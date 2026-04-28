#define TINYOBJLOADER_DISABLE_FAST_FLOAT
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <map>
#include <vector>
#include <cstring>
#include <limits>
#include <cassert>
#include <ranges>
#include <fstream>
#include <array>
#include <optional>
#include <filesystem>
#include <set>
#include "main.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/* QUICK NOETS!!!!!
GLFW creates the window,
Context opens Vulkan globally,
Instance starts a Vulkan session,
PhysicalDevice chooses the GPU,
Device opens that GPU for use,
graphicsQueue gives me a channel to submit GPU work.
The physical device exposes queue families (hardware capabilities), and when I create logical device you request queues from those families
*/

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface
#include <GLFW/glfw3.h>

constexpr uint32_t WIDTH = 800;  // measured in screen coordinates, but swap chain extent must be specified in pixels, Vulkan works with pixels
// if we are using a high DPI display (like Apple's Retina display), screen coordinates don’t correspond to pixels
// due to the higher pixel density, the resolution of the window in pixel will be larger than the resolution in screen coordinates
constexpr uint32_t HEIGHT = 600;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct UniformBufferObject { // a container of matrices we send from CPU to GPU (shader) every frame, global data shared across all vertices
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct Vertex {
    glm::vec3 pos;  // CPU memory layout question, laid out in RAM
    glm::vec3 normal;

    // | pos.x | pos.y | pos.z | normal.x | normal.y | normal.z |
    // | 4B | 4B | 4B | 4B | 4B | 4B |

    static vk::VertexInputBindingDescription getBindingDescription() { // I have a vertex buffer at binding 0. Each vertex is sizeof(Vertex) bytes, and we should step through it one vertex at a time
        return vk::VertexInputBindingDescription{
            .binding = 0, // When Vulkan looks at buffer slot 0, interpret it like THIS
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription{ // Inside each vertex from buffer binding 0, go to the pos field, read 3 floats, and send them to shader input location 0
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, pos)
            },
            vk::VertexInputAttributeDescription{ // Inside each vertex from buffer binding 0, go to the normal field, read 3 floats, and send them to shader input location 1
                .location = 1,  // send to shader layout(location = 1)
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, normal) // Give me the number of bytes from the start of Vertex to the field normal, go to byte 12 in the struct
            }
        };
    }
};

struct MeshInput {
    std::vector<Vertex>* vertices = nullptr;
    std::vector<uint32_t>* indices = nullptr;
    std::vector<int>* anchors = nullptr;
    glm::vec3 delta{ 0.0f, 0.0f, 0.0f };
};

static MeshInput* g_meshInput = nullptr;


//--------------------------------------------------------------------------------- THE GUY
class MeshApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    // When the shader asks for location 0, read from pos
    // When the shader asks for location 1, read from normal
// raii for standalone owning objects
private:
    GLFWwindow* window = nullptr;
    vk::raii::Context context;  // global Vulkan entry point
    vk::raii::Instance instance = nullptr;  // Vulkan session
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;  // actual useable connection to that GPU
    vk::PhysicalDeviceFeatures deviceFeatures;  // these are retrieved from device, so no raii
    vk::Queue queue = nullptr;
    uint32_t queueIndex = ~0u;
    vk::raii::SurfaceKHR surface = nullptr;  // usage is platform-agnostic but creation is not because it is dependent on window system details. The platform-specific addition to the extension is included in the list from glfwGetRequiredInstanceExtensions
    vk::Extent2D swapChainExtent;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;

    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffer commandBuffer = nullptr;

    vk::raii::Semaphore presentCompleteSemaphore = nullptr;
    vk::raii::Semaphore renderFinishedSemaphore = nullptr;
    vk::raii::Fence drawFence = nullptr;

    /********/
    std::vector<Vertex> vertices; // CPU copy of mesh vertices, later deformation modifies vertices, editable model data
    std::vector<uint32_t> indices; // CPU copy of triangle connectivity, GPU upload reads from vertices and indices

    // The GPU cannot directly read our C++ vector, GPU-side storage, GPU-visible Vulkan objects
    // We must copy that data into Vulkan buffers
    vk::raii::Buffer vertexBuffer = nullptr; // must create Vulkan buffers and copy data into them, current GPU copy used for drawing
    vk::raii::DeviceMemory vertexBufferMemory = nullptr; // actual memory allocation backing those buffers

    /*
    user drags vertex region
    CPU updates positions in vertices
    then you re-upload updated vertex data to vertexBuffer
    */

    // Keep mesh data on the CPU in vectors, but since GPU cannot render directly from C++ vectors, must create Vulkan objects to upload the mesh to the GPU

    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    vk::raii::Buffer uniformBuffer{nullptr};
    vk::raii::DeviceMemory uniformBufferMemory{nullptr};

    vk::raii::DescriptorSetLayout descriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool descriptorPool{ nullptr };
    vk::raii::DescriptorSets descriptorSets{ nullptr };

    vk::raii::Image depthImage{ nullptr };
    vk::raii::DeviceMemory depthImageMemory{ nullptr };
    vk::raii::ImageView depthImageView{ nullptr };

    //-------------------------------------------------------------------------------------

    bool isDragging = false;
    int selectedVertex = -1;
    double mouseX = 0.0;
    double mouseY = 0.0;

    glm::vec3 dragStartWorldPos{};
    glm::vec3 originalVertexPos{};

    std::vector<glm::vec3> draggedVertexOriginalPositions;

    bool vertexBufferDirty = false;
    std::vector<int> draggedVertexGroup;

    struct Ray {
        glm::vec3 origin;
        glm::vec3 dir;
    };

    // MESSAGE PASSING GUYS!!!!!!!!!!!!! TODO, CHECK!!!!!!!!!!!!!!!!!!!!!!
    // Current cursor displacement from the drag start point
    glm::vec3 lastDragDelta{ 0.0f, 0.0f, 0.0f };
    // Original position of every mesh vertex before this drag started
    std::vector<glm::vec3> preDragAllPositions;
    // Mesh adjacency graph
    // meshNeighbors[i] = list of vertices connected to vertex i
    std::vector<std::vector<int>> meshNeighbors;
    // Current propagated displacement for every vertex
    std::vector<glm::vec3> displacement;
    // Temporary array used during 1 message-passing update
    std::vector<glm::vec3> nextDisplacement;
    // isAnchor[i] = 1 means vertex i is directly controlled by cursor drag
    std::vector<int> isAnchor;
    // True only during active drag
    bool pregelDragActive = false;

    //-------------------------------------------------------------------------------------

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) { // choose the correct GPU memory type index for your buffer allocation
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties(); // get all memory types
        /*
        gives memoryTypeCount
        memoryTypes[i]
        */

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && // Check if the i-th bit of typeFilter is 1 (true)
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i; // if memory type i is allowed AND it has the properties I want, then pick it
            }
        }

        throw std::runtime_error("failed to find suitable memory type");
    }

    void createUniformBuffer() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            uniformBuffer,
            uniformBufferMemory
        );
    }

    UniformBufferObject buildCurrentUBO() const {
        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 3.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            swapChainExtent.width / (float)swapChainExtent.height,
            0.1f,
            10.0f
        );

        ubo.proj[1][1] *= -1.0f;
        return ubo;
    }

    void updateUniformBuffer() {
        UniformBufferObject ubo = buildCurrentUBO();
        void* data = uniformBufferMemory.mapMemory(0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        uniformBufferMemory.unmapMemory();
    }

    Ray makeMouseRay(double mx, double my) const {
        UniformBufferObject ubo = buildCurrentUBO();

        float x = 2.0f * static_cast<float>(mx) / static_cast<float>(swapChainExtent.width) - 1.0f;
        float y = 2.0f * static_cast<float>(my) / static_cast<float>(swapChainExtent.height) - 1.0f;

        glm::mat4 invMVP = glm::inverse(ubo.proj * ubo.view * ubo.model);

        glm::vec4 nearPoint = invMVP * glm::vec4(x, y, 0.0f, 1.0f);
        glm::vec4 farPoint = invMVP * glm::vec4(x, y, 1.0f, 1.0f);

        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;

        Ray ray;
        ray.origin = glm::vec3(nearPoint);
        ray.dir = glm::normalize(glm::vec3(farPoint - nearPoint));
        return ray;
    }

    int pickVertex(const Ray& ray) const {
        float bestDistance = 0.06f;
        int bestIndex = -1;

        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            glm::vec3 toVertex = vertices[i].pos - ray.origin;
            float t = glm::dot(toVertex, ray.dir);

            if (t < 0.0f) {
                continue;
            }

            glm::vec3 closestPoint = ray.origin + t * ray.dir;
            float distanceToRay = glm::length(vertices[i].pos - closestPoint);

            if (distanceToRay < bestDistance) {
                bestDistance = distanceToRay;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    bool intersectRayWithPlane(
        const Ray& ray,
        const glm::vec3& planePoint,
        const glm::vec3& planeNormal,
        glm::vec3& hitPoint
    ) const {
        float denom = glm::dot(ray.dir, planeNormal);
        if (glm::abs(denom) < 1e-6f) {
            return false;
        }

        float t = glm::dot(planePoint - ray.origin, planeNormal) / denom;

        if (t < 0.0f) {
            return false;
        }

        hitPoint = ray.origin + t * ray.dir;
        return true;
    }

    void buildDraggedVertexGroup(const glm::vec3& seedPos) {
        draggedVertexGroup.clear();
        draggedVertexOriginalPositions.clear();

        float radius = 0.1f;   // TUNE THIS ARBITRARY!!!!!

        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            if (glm::length(vertices[i].pos - seedPos) < radius) {
                draggedVertexGroup.push_back(i);
                draggedVertexOriginalPositions.push_back(vertices[i].pos);
            }
        }
    }

    void uploadVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        void* data = vertexBufferMemory.mapMemory(0, bufferSize);
        std::memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        vertexBufferMemory.unmapMemory();
    }

    void createBuffer(vk::DeviceSize size, // creates a buffer, allocates compatible GPU memory, and binds them so the buffer actually stores data the GPU can use
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        buffer = vk::raii::Buffer(device, bufferInfo);  // created an empty box with a label on it, not attached to memory yet

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements(); // how much memory do I need

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*bufferMemory, 0); // * Attach this chunk of memory to this buffer

        /*
        vk::Buffer
            bound to
            vk::DeviceMemory
        Now the GPU can:
        read from it
        write to it
        use it in pipelines
        */
    }

    vk::Format findDepthFormat() {
        return vk::Format::eD32Sfloat;
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        vk::Extent3D extent{
            swapChainExtent.width,
            swapChainExtent.height,
            1
        };

        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = depthFormat,
            .extent = extent,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment
        };

        depthImage = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memReq = depthImage.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        };

        depthImageMemory = vk::raii::DeviceMemory(device, allocInfo);
        depthImage.bindMemory(*depthImageMemory, 0);

        vk::ImageViewCreateInfo viewInfo{
            .image = *depthImage,
            .viewType = vk::ImageViewType::e2D,
            .format = depthFormat,
            .subresourceRange = {
                vk::ImageAspectFlagBits::eDepth,
                0,1,0,1
            }
        };

        depthImageView = vk::raii::ImageView(device, viewInfo);
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            vertexBuffer,
            vertexBufferMemory
        );

        void* data = vertexBufferMemory.mapMemory(0, bufferSize);
        std::memcpy(data, vertices.data(), static_cast<size_t>(bufferSize)); // COPY
        vertexBufferMemory.unmapMemory();
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            indexBuffer,
            indexBufferMemory
        );

        void* data = indexBufferMemory.mapMemory(0, bufferSize);
        std::memcpy(data, indices.data(), static_cast<size_t>(bufferSize)); // COPY
        indexBufferMemory.unmapMemory();
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------

    void createDescriptorSetLayout() { // in descriptor set layout 0, binding 0 will hold exactly one uniform buffer, visible to the vertex shader
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding
        };

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize
        };

        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSet() { // creating one descriptor set from that layout.
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &*descriptorSetLayout
        };

        descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
    }

    void updateDescriptorSet() { // Descriptor set 0, binding 0, should use uniformBuffer
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = *uniformBuffer,
            .offset = 0,
            .range = sizeof(UniformBufferObject)
        };

        vk::WriteDescriptorSet descriptorWrite{
            .dstSet = *descriptorSets[0],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &bufferInfo
        };

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }

    //---------------------------------------------------------------------------------------------------

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
            "../../../models/12683_hand_v1_FINAL.obj")) {
            throw std::runtime_error(warn + err);
        }

        vertices.clear();
        indices.clear();

        std::map<int, uint32_t> objVertexToLocalVertex;

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                int objVertexID = index.vertex_index;

                auto it = objVertexToLocalVertex.find(objVertexID);

                if (it == objVertexToLocalVertex.end()) {
                    Vertex vertex{};

                    vertex.pos = {
                        attrib.vertices[3 * objVertexID + 0],
                        attrib.vertices[3 * objVertexID + 1],
                        attrib.vertices[3 * objVertexID + 2]
                    };

                    if (!attrib.normals.empty() && index.normal_index >= 0) {
                        vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2]
                        };
                    }
                    else {
                        vertex.normal = { 0.0f, 0.0f, 1.0f };
                    }

                    uint32_t newLocalID = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                    objVertexToLocalVertex[objVertexID] = newLocalID;

                    indices.push_back(newLocalID);
                }
                else {
                    indices.push_back(it->second);
                }
            }
        }

        glm::vec3 minPos(std::numeric_limits<float>::max());
        glm::vec3 maxPos(-std::numeric_limits<float>::max());

        for (const auto& v : vertices) {
            minPos = glm::min(minPos, v.pos);
            maxPos = glm::max(maxPos, v.pos);
        }

        glm::vec3 center = 0.5f * (minPos + maxPos);
        glm::vec3 extent = maxPos - minPos;
        float maxExtent = std::max(extent.x, std::max(extent.y, extent.z));

        std::cout << "Loaded unique vertices: " << vertices.size() << "\n";
        std::cout << "Loaded triangle indices: " << indices.size() << "\n";

        for (auto& v : vertices) {
            v.pos = (v.pos - center) / maxExtent;
        }
    }

    /*
    To use any vk::Image, including those in the swap chain, in the render pipeline we have to create a vk::raii::ImageView object. 
    An image view is quite literally a view into an image. It describes how to access the image and which part of the image to access, 
    for example, if it should be treated as a 2D texture depth texture without any mipmapping levels
    */

    const std::vector<char const*> validationLayers{
        "VK_LAYER_KHRONOS_validation"
    };

    std::vector<const char*> requiredDeviceExtension = {  // enable GPU capabilities
        vk::KHRSwapchainExtensionName      // "VK_KHR_swapchain" extension, required for presenting rendered images to the window
    };  // The swap chain is essentially a queue of images that are waiting to be presented to the screen, how exactly the queue works and the conditions for presenting an image from the queue depend on how the swap chain is set up. 
        // synchronize the presentation of images with the refresh rate of the screen.

    void createInstance() {  // Create a Vulkan instance using these layers and extensions
        constexpr vk::ApplicationInfo appinfo{
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14
        };

        auto requiredExtensions = getRequiredInstanceExtensions();   // GLFW tells Vulkan what instance extensions are required for this OS/window system
        // KIV for surfaces, returns pointer to array of C strings, const char**
        // instance extensions deal with window system integration, they extend the Vulkan instance
        auto extensionProperties = context.enumerateInstanceExtensionProperties();

        auto unsupportedPropertyIt =
            std::ranges::find_if(requiredExtensions,
                [&extensionProperties](auto const& requiredExtension) {
                    return std::ranges::none_of(extensionProperties,
                        [requiredExtension](auto const& extensionProperty) { 
                            return strcmp(extensionProperty.extensionName, requiredExtension) == 0; 
                        });
                });
        if (unsupportedPropertyIt != requiredExtensions.end())
        {
            throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
        }

        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        auto layerProperties = context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIt = std::ranges::find_if(requiredLayers,   // returns an iterator to the first element in the range for which the predicate returns true. If no such element is found, the function returns the end_iterator
            [&layerProperties](auto const& requiredLayer) {
                return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) {  // Returns true if the predicate returns false for every single element in the range
                        return strcmp(requiredLayer, layerProperty.layerName) == 0;
                    });
            });

        if (unsupportedLayerIt != requiredLayers.end()) {
            throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
        }

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appinfo,  // metadata
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),   // How many validation layers to enable, usually 1 if debug, 0 if build
            .ppEnabledLayerNames = requiredLayers.data(),   // Pointer to an array of C strings (layer names)
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),   // How many instance extensions to enable
            .ppEnabledExtensionNames = requiredExtensions.data()
        };

        instance = vk::raii::Instance(context, createInfo);  // omitted allocator (default)
        // This app is using Vulkan with these enabled instance extensions/layers, now we have a Vulkan "session"
        // Before, we needed global / loader-discoverable facts, that were to be queried without having yet created an instance
        // Context (has global Vulkan entry functions) talks to the loader to discover what’s available before instance creation.
        // Instance talks through the loader / driver to discover GPUs and do instance - scoped work after you’ve chosen extensions / layers.
    }

    void initWindow() {  // create window, program talks to GLFW, GLFW talks to the OS
        if (!glfwInit()) {
            throw std::runtime_error("glfwInit failed");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void createSurface() {
        VkSurfaceKHR _surface;  // this is a C API object, need to promote to C++ wrapper
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface); // wrap it, destroying is taken care of by Vulkan RAII
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::ranges::find_if(devices,
            [&](auto const& device) {
                auto queueFamilies = device.getQueueFamilyProperties();
                bool isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                const auto qfpIter = std::ranges::find_if(queueFamilies,
                    [](vk::QueueFamilyProperties const& qfp) {
                        return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);   // Does the graphics bit exist in this bitmask? If result not zero, graphics supported, convert 0 into correct type
                    });

                isSuitable = isSuitable && (qfpIter != queueFamilies.end());  // if no queue supports graphics, there are different types of queues that originate from different queue families, and each family of queues allows only a subset of commands
                auto extensions = device.enumerateDeviceExtensionProperties();
                bool found = true;
                for (auto const& extension : requiredDeviceExtension) {    // checked if GPU supported swapchain extension in principle
                    auto extensionIter = std::ranges::find_if(extensions, [extension](auto const& ext) {return strcmp(ext.extensionName, extension) == 0; });
                    found = found && extensionIter != extensions.end();
                }
                isSuitable = isSuitable && found;
                if (isSuitable) {
                    physicalDevice = device;
                }
                return isSuitable;
            });
        if (devIter == devices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    uint32_t findQueueFamilies(vk::raii::PhysicalDevice physicalDevice) {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
            physicalDevice.getQueueFamilyProperties();

        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
                // found a queue family that supports both graphics and present to window surface
                queueIndex = qfpIndex;
                break;
            }
        }

        if (queueIndex == ~0u) {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        return queueIndex;
    }

    void createLogicalDevice() {   // set up logical device to interface with physical device, create a logical device for this GPU, with these queues, these features, and these device extensions
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        // Each element in the vector contains things like: queueFlags (what work queueFamily can perform), queueCount, timestampValidBits, minImageTransferGranularity
        uint32_t queueIndex = findQueueFamilies(physicalDevice); // search through queue families, find the first one that supports graphics
        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
            .queueFamilyIndex = queueIndex, 
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };
        //  can create all the command buffers on multiple threads and then submit them all at once on the main thread with a single low-overhead call, so usually one queue per queue family
        //  To use newer features, you need to explicitly request them during device creation
        //  Create a chain of feature structures
        //  When creating the logical device, I also want these newer GPU features enabled
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
            {},                               // vk::PhysicalDeviceFeatures2 (empty for now)
            {.dynamicRendering = true },      // Enable dynamic rendering from Vulkan 1.3
            {.extendedDynamicState = true }   // Enable extended dynamic state from the extension
        };
        // The vk::StructureChain template automatically connects these structures together by setting up the pNext pointers between them. When we create the logical device later, we’ll pass a pointer to the first structure in this chain, which will allow Vulkan to see all the features we want to enable

        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQueueCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
            .ppEnabledExtensionNames = requiredDeviceExtension.data()  // all structures in the feature chain are connected, can follow the pNext pointers
            // Vulkan processes each structure in the chain and enables the requested features during device creation
        };

        device = vk::raii::Device(physicalDevice, deviceCreateInfo); // physical device to interface with, and usage info we just specified
        queue = device.getQueue(queueIndex, 0);  // Which exact queue-family index should I request from Vulkan?
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats) {  // each vk::SurfaceFormatKHR has format and colorSpace member (SRGB)
        const auto formatIt = std::ranges::find_if(availableFormats, [](const auto &format) {
            return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
            });
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0]; // settle with first format if all else fails
    }

    vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes) {
        assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { 
            return presentMode == vk::PresentModeKHR::eFifo; 
            }));
        return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) { 
            return vk::PresentModeKHR::eMailbox == value; 
            }) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    // must use glfwGetFramebufferSize to query the resolution of the window in pixel before matching it against the minimum and maximum image extent
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities) {  // decides the VkExtent2D for the swapchain
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {  // this means "You choose the swapchain resolution"
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);  // query the resolution of the window in pixel

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {  // decides how many images the swapchain should contain, triple buffering
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);  // at least 3 swapchain images
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;    // do not exceed GPU limits
        }
        return minImageCount;
    }

    void createImageViews() {
        assert(swapChainImageViews.empty());
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,  // 2d render target in most cases when we’re rendering to a screen
            .format = swapChainSurfaceFormat.format,  // how the colorspace components are configured
            .components = {
                vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity
            },
            .subresourceRange = {
                vk::ImageAspectFlagBits::eColor,
                0,  // baseMipLevel
                1,  // levelCount
                0,  // baseArrayLayer
                1   // layerCount
            }
        };
        for (auto& image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    // The swapchain step comes AFTER logical device creation, because swapchains require: surface, logical device, queue family index
    void createSwapChain() {
        // need to check three kinds of properties after checking if swap is available
        // 1. basic surface capabilities
        // 2. surface formats
        // 3. available presentation modes
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface); // query basic surface capabilities
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

        std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);
        // need to find the best possible swapchain

        // Present modes: eImmediate, eFifo, eFifoRelaxed, eMailbox

        vk::SwapchainCreateInfoKHR swapChainCreateInfo { 
            .surface = *surface,
            .minImageCount = minImageCount,
            .imageFormat = swapChainSurfaceFormat.format,
            .imageColorSpace = swapChainSurfaceFormat.colorSpace,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,   // the imageArrayLayers specifies the number of layers each image consists of. This is always 1 unless we are developing a stereoscopic 3D application
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, // the imageUsage bit field specifies what kind of operations we'll use the images in the swap chain for. Here we render directly to them, which means that they are used as color attachment. It is also possible that we render images to a separate image first to perform operations like post-processing
            .imageSharingMode = vk::SharingMode::eExclusive,  // an image is owned by one queue family at a time, and ownership must be explicitly transferred before using it in another queue family. This option offers the best performance, or eConcurrent (images can be used across multiple queue families without explicit ownership transfers.)
            // Concurrent mode requires you to specify at least two distinct queue families
            .preTransform = surfaceCapabilities.currentTransform,  // specify that a certain transform should be applied to images in the swap chain if it is supported
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,   // specifies if the alpha channel should be used for blending with other windows in the window system. We almost always want to simply ignore the alpha channel
            .presentMode = presentMode,
            .clipped = true,
            .oldSwapchain = nullptr
        };

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }


    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{ 
            .codeSize = code.size() * sizeof(char), 
            .pCode = reinterpret_cast<const uint32_t*>(code.data()) 
        };
        vk::raii::ShaderModule shaderModule { device, createInfo };
        return shaderModule;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::cout << "cwd = " << std::filesystem::current_path() << '\n';
        std::cout << "trying to open = " << filename << '\n';

        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file: " + filename);
        }

        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("../../../shaders/slang.spv"));
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ 
            .stage = vk::ShaderStageFlagBits::eVertex, 
            .module = shaderModule,  
            .pName = "vertMain" 
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ 
            .stage = vk::ShaderStageFlagBits::eFragment, 
            .module = shaderModule, 
            .pName = "fragMain" 
        };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        
        // pVertexBindingDescriptions and pVertexAttributeDescriptions (NOTE!)
        // binding description = how to step from one vertex to the next
        // attribute descriptions = how to decode one vertex's fields and map them to shader inputs
        // pipeline knows how vertex-buffer bytes should be interpreted
        // The shader module is already created from SPIR-V, and the pipeline also stores: vertex shader stage, fragment shader stage
        // pipeline says: if something is bound at slot 0, here is how to read it
        // draw command says: here is the actual buffer to use for slot 0
        // GPU does: for each vertex index:
        //     find the bound buffer at slot 0
        //     compute address using stride
        //     read attributes using offset / format
        //     feed them into vertex shader inputs
        // The graphics pipeline does not itself read the vertices immediately.
        // It stores the rules for how to read them.
        // The actual reading happens at draw time, when:
        //    the pipeline is bound
        //    the vertex buffer is bound
        //    draw() is called
        /*
            CPU vertices
                ↓ copy
            GPU vertex buffer (raw bytes)

            binding description
                = how to move vertex-to-vertex

            attribute descriptions
                = how to decode fields inside one vertex

            graphics pipeline
                = stores these decoding rules + shader 

            bindVertexBuffers(...)
                = supplies the actual buffer object

            draw(...)
                = GPU finally reads vertices and runs the pipeline
        */
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ 
            .topology = vk::PrimitiveTopology::eTriangleList
        }; // what kind of geometry will be drawn from the vertices

        // viewport basically describes the region of the framebuffer that the output will be rendered to.This will almost always be(0, 0) to(width, height)
        vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f };
        // we intend to render across the whole current swapchain image
        // take NDC coordinates in [-1, 1]
        // and map them into pixel coordinates :
        // x → [0, width]
        // y → [0, height]
        // triangle is now expressed relative to the swapchain extent
        // after rasterization, we get fragments for covered pixels
        // image is not updated yet until rasterization generates fragments and the surviving fragments are written into the image

        vk::Rect2D( vk::Offset2D(0, 0), swapChainExtent);
        // Scissor is applied after rasterization, before writing
        // It says:
        // "only pixels inside this rectangle are allowed to be written"

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();  // required to specify the data at runtime

        // fill in the struct explicitly
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };  // only need to specify count since we are using dynamic state, actual viewport and scissors set up at drawing time

        // Culling = removing triangles based on how they are oriented relative to YOUR VIEW
        // Rasterization happens AFTER culling
        /*
        The rasterizer takes the geometry shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader. It also performs depth testing, face culling and the scissor test, and it can be configured to output fragments that fill entire polygons or just the edges (wireframe rendering)
        */

        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        vk::PipelineRasterizationStateCreateInfo rasterizer { 
            .depthClampEnable = vk::False, // if set to VK_TRUE, then fragments that are beyond the near and far planes are clamped to them as opposed to discarding them
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill, //  fill the area of the polygon with fragments
            .cullMode = vk::CullModeFlagBits::eBack, // determines the type of face culling to use, can disable culling, cull the front faces, cull the back faces or both
            .frontFace = vk::FrontFace::eClockwise, // specifies the vertex order for the faces to be considered front-facing and can be clockwise or counterclockwise.
            .depthBiasEnable = vk::False,
            .depthBiasSlopeFactor = 1.0f, 
            .lineWidth = 1.0f // describes the thickness of lines in terms of number of fragments
        };

        // anti-aliasing = reducing jagged edges caused by sampling a continuous scene onto discrete pixels
        // The VkPipelineMultisampleStateCreateInfo struct configures multisampling, which is one of the ways to perform antialiasing, KIV!
        vk::PipelineMultisampleStateCreateInfo multisampling{ 
            .rasterizationSamples = vk::SampleCountFlagBits::e1, 
            .sampleShadingEnable = vk::False
        };

        /*
        Vertex data is read from buffers → processed by the vertex shader → grouped into primitives during primitive assembly → 
        optionally modified by tessellation and geometry shaders
        */

        // KIV depth and stencil buffers
        // fragment is (position, interpolated attributes, depth, etc.)
        // MANY fragments may target the SAME pixel → GPU resolves conflicts using: depth test, blending
        // A swapchain has many images → for each frame, we take one of those images → we put it into a framebuffer as a color attachment → we render into that image

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA   // write all channels
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False, // no bitwise mixing
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        // Uniforms are data you pass from CPU → shader at runtime without recreating the pipeline
        /*
        Shaders are compiled into the pipeline. So we cannot hardcode changing values like:
            camera position
            transformation matrices
            time
            lighting
        */

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
        };

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // A descriptor pool is where descriptor sets are allocated from.

        // END OF FIXED FUNCTION STATE

        // When this pipeline runs, what kind of images (attachments) will it render into?
        // Dynamic rendering simplifies the rendering process by eliminating the need for render pass and framebuffer objects. Instead, we can specify the color, depth, and stencil attachments directly when we begin rendering
        vk::Format depthFormat = findDepthFormat();

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
            .depthAttachmentFormat = depthFormat   // 🔥 ADD THIS
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess
        };

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .pNext = &pipelineRenderingCreateInfo,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,   // 🔥 ADD THIS
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .renderPass = nullptr,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };
        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }
    // The GPU has different queue families for different types of work (graphics / compute / transfer / present)
    // Command buffers are created for a specific queue family
    // A command pool allocates command buffers tied to that queue family
    void createCommandPool() { // CPU records instructions → puts them in a list → sends list to GPU
        //-------------------------------------------------------------------------------------------------------------------------
        // in Vulkan 1.3, we can now render directly to image views without creating framebuffers or render passes
        // Commands in Vulkan, like drawing operations and memory transfers, are not executed directly using function calls. You have to record all the operations you want to perform in command buffer objects

        // We will be recording a command buffer every frame
        // Command buffers are executed by submitting them on one of the device queues, like the graphics and presentation queues we retrieved. Each command pool can only allocate command buffers that are submitted on a single type of queue
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, // Allow command buffers to be rerecorded with new commands individually, without this flag they all have to be reset together
            .queueFamilyIndex = queueIndex // bind it
        };

        commandPool = vk::raii::CommandPool(device, poolInfo); // We're going to record commands for drawing, which is why we've chosen the graphics queue family
    }

    void recordCommandBuffer(uint32_t imageIndex) { // writes commands we want to execute into a command buffer, our chosen image is imageIndex
        // not now
        commandBuffer.reset();
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer.begin(beginInfo);
        //-------------------------------------------------------------------------------------------------------------------
        // transition swapchain image so it can be rendered into, transition is itself a GPU command recorded into the command buffer
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);  // clear the color attachment to opaque black before drawing

        vk::RenderingAttachmentInfo depthAttachment{
            .imageView = *depthImageView,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = vk::ClearValue{ vk::ClearDepthStencilValue{1.0f, 0} }
        };

        // A rendering pass is a period of GPU execution where you write fragments into attachments (images)
        vk::RenderingAttachmentInfo attachmentInfo = { // This describes one attachment used during this rendering pass, how to use image during rendering
            .imageView = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, // must match the earlier layout transition
            .loadOp = vk::AttachmentLoadOp::eClear,        // "When rendering starts, CLEAR the image"
            .storeOp = vk::AttachmentStoreOp::eStore,      // "After rendering, KEEP the result"
            .clearValue = clearColor
        };

        vk::RenderingInfo renderingInfo = {     // the whole rendering setup
            .renderArea = {       // "Render over THIS region of the image"
                .offset = { 0, 0 }, 
                .extent = swapChainExtent 
            },
            .layerCount = 1,            // "How many layers (for array/3D images)"
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo,
            .pDepthAttachment = &depthAttachment
        };

        vk::ImageMemoryBarrier depthBarrier{
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .image = *depthImage,
            .subresourceRange = {
                vk::ImageAspectFlagBits::eDepth,
                0,1,0,1
            }
        };

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {},
            nullptr,
            nullptr,
            depthBarrier
        );

        commandBuffer.beginRendering(renderingInfo); // "GPU, start rendering into the attachments I just described"

        // This is a graphics pipeline (not compute)
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline); // "Use THESE rules to turn vertices into pixels", "Use THIS pipeline to process all upcoming draw calls"
        /* graphics pipeline contains
        - vertex shader
        - fragment shader
        - how vertices are read
        - how triangles are formed
        - rasterization rules
        - blending rules
        */
        // MUST BIND PIPELINE BEFORE DRAWING
        // We've now told Vulkan which operations to execute in the graphics pipeline and which attachment to use in the fragment shader
        // set dynamic components before issuing draw command

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout,
            0,
            *descriptorSets[0], // binding table for the shader
            nullptr
        );

        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

        vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, *vertexBuffer, offsets);
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);

        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        /*
        read indices from the bound index buffer
        for each index, fetch the corresponding vertex from the bound vertex buffer
        assemble triangles
        rasterise them

        vertexCount: Even though we don’t have a vertex buffer, we technically still have 3 vertices to draw. "PROCESS 3 VERTICES"
        instanceCount : Used for instanced rendering, use 1 if you’re not doing that. "DRAW THIS ONCE"
        firstVertex : Used as an offset into the vertex buffer, defines the lowest value of SV_VertexId. "START VERTEX IDS FROM 0"
        firstInstance : Used as an offset for instanced rendering, defines the lowest value of SV_InstanceID. "START INSTANCE INDEX FROM 0"
        */

        commandBuffer.endRendering();
        
        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},                                           
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,    
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        //-------------------------------------------------------------------------------------------------------------------
        commandBuffer.end();
    }

    /*
    Vulkan image also has a current usage/layout state. For a swapchain image, a typical frame does this:
    present engine was using image for presentation
    : transition to color-attachment layout
    : render into it
    : transition back to present layout
    : present it
    */

    void transition_image_layout(
        uint32_t imageIndex,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags2 srcAccessMask,
        vk::AccessFlags2 dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask
    ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
    }

    void createCommandBuffer() {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,  // can be submitted to a queue for execution, but cannot be called from other command buffers, secondary would be useful to reuse common operations
            .commandBufferCount = 1  // only allocating one command buffer
        };
        commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front()); // transfer ownership of this command buffer into my commandBuffer variable
    }

    // WUICK NOTE!!!!!!!!!
    /* Rendering a frame in Vulkan:
        wait for previous frame to finish
        acquire an image from the swapchain
        record a command buffer which draws the scene onto that image
        submit the recorded command buffer
        present the swapchain image
    */
    //  synchronisation of execution on the GPU is explicit, functions may return before the operation has finished for various API calls

    // if each of the operations depends on the previous one finishing, we need to explore which primitives we can use to achieve the desired ordering.
    // binary semaphores: vkQueueSubmit(work: A, signal: S, wait: None), vkQueueSubmit(work: B, signal: None, wait: S)
    // waiting only happens on the GPU. The CPU continues running without blocking. To make the CPU wait, we need a different synchronization primitive, fences
    /*
    // enqueue A, start work immediately, signal F when done
    vkQueueSubmit(work: A, fence: F)
    vkWaitForFence(F) // blocks execution until A has finished executing, BLOCK HOST EXECUTION
    save_screenshot_to_disk() // can't run until the transfer has finished
    */
    // Becase we re-record the command buffer every frame, we cannot record the next frame’s work to the command buffer until the current frame has finished executing
    
    /*
    acquireNextImage(...)
       returns imageIndex
       signals presentCompleteSemaphore

    submit draw:
       waits on presentCompleteSemaphore
       renders into that image

    GPU finishes
       signals renderFinishedSemaphore

    present:
       waits on renderFinishedSemaphore
       shows image
    */

    void drawFrame() {
        // 1. CPU-GPU sync
        auto fenceResult = device.waitForFences(*drawFence, vk::True, UINT64_MAX); // CPU: Pause CPU execution HERE until previous frame is done by GPU, drawFence is signaled by GPU
        // vk::Result::eSuccess : fence was signaled (normal case)
        // vk::Result::eTimeout : timeout happened(only if we DIDN'T use UINT64_MAX)
        // 2. Get next image, signal presentCompleteSemaphore when image is safe

        if (vertexBufferDirty) {
            uploadVertexBuffer();
            vertexBufferDirty = false;
        }

        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore, nullptr); // signal when image becomes available, I DON'T want GPU to start rendering immediately
        
        updateUniformBuffer();
        // now I have image
        recordCommandBuffer(imageIndex);
        device.resetFences(*drawFence);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput); // GPU can run earlier stages, BUT must WAIT before writing to framebuffer. Vertex shader can run early. fragment output must wait
        // describes ONE GPU submission to a queue, KIV: queue.submit(submitInfo, drawFence)
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*presentCompleteSemaphore, // Do NOT start rendering until swapchain image is ready
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer, //  the actual work GPU will execute
            .signalSemaphoreCount = 1, // after GPU finish then signal this semaphore
            .pSignalSemaphores = &*renderFinishedSemaphore 
        };

        queue.submit(submitInfo, *drawFence);

        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex 
        };

        result = queue.presentKHR(presentInfoKHR);
    }

    void createSyncObjects() {
        presentCompleteSemaphore = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo()); // Image is ready and safe to use, can start rendering
        renderFinishedSemaphore = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo()); // GPU finished rendering, now safe to PRESENT
        drawFence = vk::raii::Fence(device, { .flags = vk::FenceCreateFlagBits::eSignaled } ); // Create a fence starting in the "signaled" state, submit GPU work (with fence attached), GPU finishe, then signals fence
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();

        loadModel(); // loadModel() fill CPU ARRAYS
        createVertexBuffer();
        createIndexBuffer();

        createUniformBuffer();
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSet();
        updateDescriptorSet();

        createDepthResources();

        createGraphicsPipeline();
        createCommandPool();
        createCommandBuffer();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            glfwGetCursorPos(window, &mouseX, &mouseY);
            handleMouseInteraction(); // RUN INTERACTION LOGIC
            drawFrame();
        }
    }

    void cleanup() {
        device.waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    std::vector<const char*> getRequiredInstanceExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        return extensions;
    }

    void handleMouseInteraction() {
        bool leftPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        Ray ray = makeMouseRay(mouseX, mouseY);
        if (leftPressed && !isDragging) {
            selectedVertex = pickVertex(ray);
            if (selectedVertex != -1) {
                isDragging = true;
                preDragAllPositions.clear();
                preDragAllPositions.reserve(vertices.size());
                for (const auto& v : vertices) {
                    preDragAllPositions.push_back(v.pos);
                }

                originalVertexPos = vertices[selectedVertex].pos;
                buildDraggedVertexGroup(originalVertexPos);

                glm::vec3 hitPoint; // 3D point on drag plane
                bool ok = intersectRayWithPlane(ray, originalVertexPos, glm::vec3(0.0f, 0.0f, 1.0f), hitPoint);
                if (ok) { // if hit successfully, only update the mesh if we successfully found a valid 3D mouse position
                    dragStartWorldPos = hitPoint; // SAVED
                }
                else {
                    isDragging = false;
                    selectedVertex = -1;
                    draggedVertexGroup.clear();
                }
            }
        } else if (leftPressed && isDragging) {
            glm::vec3 hitPoint;
            bool ok = intersectRayWithPlane(ray, originalVertexPos, glm::vec3(0.0f, 0.0f, 1.0f), hitPoint);
            if (ok) {
                lastDragDelta = hitPoint - dragStartWorldPos;
                for (size_t k = 0; k < draggedVertexGroup.size(); k++) {
                    int idx = draggedVertexGroup[k];
                    vertices[idx].pos = draggedVertexOriginalPositions[k] + lastDragDelta;
                }
                vertexBufferDirty = true;
            }
        }
        else if (!leftPressed && isDragging) {
            std::cout << "[rank " << get_worker_id() << "] mouse released, entering meshPregel\n";
            meshPregel(lastDragDelta, draggedVertexGroup);
            std::cout << "[rank " << get_worker_id() << "] meshPregel returned\n";
            vertexBufferDirty = true;
            isDragging = false;
            selectedVertex = -1;
            draggedVertexGroup.clear();
            draggedVertexOriginalPositions.clear();
        }
    }

    // Vulkan mesh, dragged anchor vertices, cursor displacement
    class MeshPregelVertex : public BVertex<MeshValue, int, MeshMessage, DefaultHash, BaseCombiner<MeshMessage>, BaseAggregator<void> > {
    public:
        typedef MeshValue ValueType;
        typedef int EdgeType;
        typedef MeshMessage MessageType;
        typedef DefaultHash HashType;
        typedef BaseCombiner<MeshMessage> CombinerType;
        typedef BaseAggregator<void> AggregatorType;

        MeshPregelVertex() {}
        MeshPregelVertex(VertexID id) : BVertex<MeshValue, int, MeshMessage, DefaultHash, BaseCombiner<MeshMessage>, BaseAggregator<void> >(id) {}
        MeshPregelVertex(VertexID id, MeshValue &value) : BVertex<MeshValue, int, MeshMessage, DefaultHash, BaseCombiner<MeshMessage>, BaseAggregator<void> >(id, value) {}

        bool operator<(const MeshPregelVertex& rhs) const {
            return this->id() < rhs.id();
        }

        bool operator==(const MeshPregelVertex& rhs) const {
            return this->id() == rhs.id();
        }

        bool operator!=(const MeshPregelVertex& rhs) const {
            return this->id() != rhs.id();
        }

        void compute(std::vector<MeshMessage> &messages) {
            if (!value().anchor && !messages.empty()) {
                float sx = 0.0f;
                float sy = 0.0f;
                float sz = 0.0f;

                for (auto& msg : messages) {
                    sx += msg.dx;
                    sy += msg.dy;
                    sz += msg.dz;
                }

                float inv = 1.0f / static_cast<float>(messages.size());
                float avgX = sx * inv;
                float avgY = sy * inv;
                float avgZ = sz * inv;
                float alpha = 0.5f; // arbitrary!!

                value().disp_x = (1.0f - alpha) * value().disp_x + alpha * avgX;
                value().disp_y = (1.0f - alpha) * value().disp_y + alpha * avgY;
                value().disp_z = (1.0f - alpha) * value().disp_z + alpha * avgZ;
            }

            MeshMessage out;
            out.dx = value().disp_x;
            out.dy = value().disp_y;
            out.dz = value().disp_z;

            for (auto& e : edges()) {
                send_message(e.target, out);
            }

            if (step_num() >= 100) {
                vote_for_halt();
            }
        }
    };

    class MeshGraphLoader {
    public:
        void set_id(int worker_id) {
            id = worker_id;
        }

        void set_buffer(GraphBuffer<MeshPregelVertex>* b) {
            buffer = b;
        }

        void load_graph() {
            MeshInput &input = *g_meshInput;
            int n = input.vertices->size();
            std::vector<int> isAnchor(n, 0);
            for (int idx : *input.anchors) {
                if (idx >= 0 && idx < n) {
                    isAnchor[idx] = 1;
                }
            }

            for (int i = 0; i < n; i++) {
                glm::vec3 p = (*input.vertices)[i].pos;
                MeshValue value;
                value.base_x = p.x;
                value.base_y = p.y;
                value.base_z = p.z;
                if (isAnchor[i]) {
                    value.disp_x = input.delta.x;
                    value.disp_y = input.delta.y;
                    value.disp_z = input.delta.z;
                    value.anchor = 1;
                } else {
                    value.disp_x = 0.0f;
                    value.disp_y = 0.0f;
                    value.disp_z = 0.0f;
                    value.anchor = 0;
                }
                buffer->add_vertex(i, value);
            }

            auto addUndirectedEdge = [&](int a, int b) { 
                int dummy = 1;
                BEdge<int> e1(b, dummy);
                BEdge<int> e2(a, dummy);
                buffer->add_edge(a, e1);
                buffer->add_edge(b, e2);
            };

            for (size_t i = 0; i + 2 < input.indices->size(); i += 3) {
                int a = static_cast<int>((*input.indices)[i]);
                int b = static_cast<int>((*input.indices)[i + 1]);
                int c = static_cast<int>((*input.indices)[i + 2]);
                addUndirectedEdge(a, b);
                addUndirectedEdge(b, c);
                addUndirectedEdge(c, a);
            }
        }
    private:
        int id = 0;
        GraphBuffer<MeshPregelVertex>* buffer = nullptr;
    };

    void meshPregel(const glm::vec3& delta, const std::vector<int>& anchors) {
        int workers = get_num_workers();
        if (workers != 1) {
            std::cerr << "ERROR: interactive meshPregel currently only works with 1 MPI rank \n";
            std::cerr << "Currently running with " << workers << " ranks \n";
            return;
        }

        MeshInput input;
        input.vertices = &vertices;
        input.indices = &indices;
        input.anchors = const_cast<std::vector<int>*>(&anchors);
        input.delta = delta;
        g_meshInput = &input;

        int num_partitions_dum = 1;
        Worker<MeshPregelVertex, MeshGraphLoader> worker;
        worker.run(num_partitions_dum);

        for (auto& part : worker.local_partitions()) {
            for (auto& v : part.vertexes()) {
                int id = v.id();

                if (id < 0 || id >= static_cast<int>(vertices.size())) {
                    continue;
                }

                MeshValue& val = v.value();

                vertices[id].pos = glm::vec3(
                    val.base_x + val.disp_x,
                    val.base_y + val.disp_y,
                    val.base_z + val.disp_z
                );
            }
        }
        vertexBufferDirty = true;
        g_meshInput = nullptr;
    }
};

int main(int argc, char** argv) {
    try { // CHECK!!!!!
        init_workers(&argc, &argv); // MPI STARTUP GO
        MPI_Barrier(MPI_COMM_WORLD);
        MeshApp app;
        app.run();
        MPI_Finalize();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}