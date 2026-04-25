struct UniformBufferObject { 
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 normal : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertMain(
    [[vk::location(0)]] float3 inPos,
    [[vk::location(1)]] float3 inNormal)
{
    VertexOutput output;

    float4 pos = float4(inPos, 1.0);
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, pos)));

    // transform normal by model matrix
    output.normal = normalize(mul((float3x3)ubo.model, inNormal));

    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target
{
    float3 normal = normalize(inVert.normal);
    float3 lightDir = normalize(float3(0.5, 0.7, 1.0));

    float diffuse = max(dot(normal, lightDir), 0.0);
    float ambient = 0.2;
    float brightness = ambient + 0.8 * diffuse;

    float3 baseColor = float3(0.8, 0.85, 0.95);
    return float4(baseColor * brightness, 1.0);
}