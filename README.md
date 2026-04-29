# Distributed Mesh Deformation on Pregel
Experimental self-implemented C++ Pregel on 3D mesh deformation demonstration. Fundamentally working but lacking optimisations on the Pregel side (not much) or the graphics side (not so sure about what in Vulkan specifically) or the algorithm side (even a simple Euler integration would probably be better than my current compute() function, which is overly simple because I want to focus on implementing Pregel here, but it would be worthwhile to read ARAP, discrete curvature, Pixar hair modelling, NVIDIA papers, for better idea of more PBR algorithms). Or it could simply be that message-passing is overkill for something "straightforward" like mesh deformation and the overhead is too much

## Important files
- CMakeProject1.cpp is the main Vulkan application and some mesh-specific Pregel logic (this does not seem like good organisation but it will suffice)
- main.h is my Pregel-style framework: vertices, edges, partitions, graph buffer, message buffer, worker loop, serialisation and other stuff
- comm.cpp is the MPI setup and communication helpers
- models/
 12683_hand_v1_FINAL.obj is a free model I downloaded online. Note that this is only .obj data, rendering is done entirely through the Vulkan API. I will not explain the entire rendering pipeline here

## Explanation of how this code flows
- User does a mouse drag on my GLFW window
- MeshApp::handleMouseInteraction() is called
- we select anchor vertices near cursor (from some casted ray intersections)
- meshPregel(delta, anchors) is called
- MeshGraphLoader builds graph from Vulkan mesh (sent to it from my Vulkan-side code in the form of a MeshInput)
- Worker runs Pregel supersteps
- MeshPregelVertex::compute() averages neighbour displacement messages
- updated displacement written back to vertices[id].pos
- vertexBufferDirty = true
- uploadVertexBuffer()
- Vulkan renders modified mesh

## Idea
The mesh is converted into a graph

For every mesh vertex:
Vertex i becomes Pregel vertex i
For every triangle (a, b, c), we have undirected edges: (a -- b), (b -- c), (c -- a)
Then the drag displacement is propagated through this graph using messages. Anchor vertices are directly controlled by the mouse drag. Non-anchor vertices update their displacement by averaging the displacement messages received from neighbouring vertices

## More explanations
### CMakeProject1.cpp (can just ignore if unfamiliar with graphics pipeline)
This is the main application file

- Vulkan window and rendering setup
- obj model loading through tiny_obj_loader
- CPU-side mesh storage: (NOTE! This is what is edited before presentation to GPU-side Vulkan buffers)
  - std::vector<Vertex> vertices
  - std::vector<uint32_t> indices
- GPU-side Vulkan buffers:
  - vertex buffer
  - index buffer
  - uniform buffer
- mouse picking and dragging logic
- mesh-specific Pregel logic: MeshPregelVertex, MeshGraphLoader, meshPregel(args...)

### main.h
Generic Pregel-like framework. Note MeshValue stores the base vertex position, current displacement and whether the vertex is an anchor
- MeshMessage is the displacement vector sent between neighbouring vertices.
- BVertex is one graph vertex
- BEdge is one graph edge
- Division isone partition containing many vertices
- GraphBuffer is temporary graph builder and graph distributor
- MessageBuffer is the big mailbox to stores outgoing and incoming messages (*)
- Worker runs supersteps
- SerialMe converts objects into bytes for MPI
- UnSerialMe reconstructs objects from  reading bytes

### comm.cpp
Just some wrapping of the MPI communication layer used by the Pregel framework here
Each superstep does
1. Every local vertex runs compute(messages)
2. Vertices update their own value
3. Vertices send messages to neighbours
4. The message buffer exchanges messages between workers
5. The worker checks whether all partitions voted to halt
6. If not halted, next superstep begins

so basically
```cpp
while not halted:
    set global step number
    reset aggregator
    for each local partition of vertices:
        compute every vertex
    sync messages
    aggregate global data
    check global halt condition
```

I also put the mpi.h filepath very local, so please change it to wherever your path is. models/12683_hand_v1_FINAL.obj needs to be downloaded, or just replace with any .obj file you want. This is what happens when you pull a subset of vertices, you can clearly see that the mesh is STRETCHING instead of just vertices being PROTRUDED straight out. This is because the message passing is working and neighbouring vertices are hearing the messages that their neighbours got displaced, so they are being dragged along as well. This is only run for a few iterations because its quite slow, so with more iterations of message passing, the entire mesh should seem like rubber and THE REST OF THE HAND WILL DRAG ALONG THE DISPLACEMENT. With only a few rounds of message passing, you can see that only the surrounding neighbour vertices of the originallly displaced vertices have started to drag along.

I need to optimise this or its just not viable at all. This needs some more significant ideas. But at least it shows that my Pregel baseline is functional
<img width="1920" height="1080" alt="Screenshot (247)" src="https://github.com/user-attachments/assets/16ef16c6-9bd6-406d-ac14-4ff0ab9dc0f1" />
