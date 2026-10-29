// Format features of block-compressed formats on this driver (blit source, storage, ...).
#include <stdio.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "fmt", 1, "fmt", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    const struct { VkFormat f; const char* name; } formats[] = {
        {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "BC1"}, {VK_FORMAT_BC3_UNORM_BLOCK, "BC3"},
        {VK_FORMAT_BC4_UNORM_BLOCK, "BC4"},      {VK_FORMAT_BC5_UNORM_BLOCK, "BC5"},
        {VK_FORMAT_BC6H_UFLOAT_BLOCK, "BC6H"},   {VK_FORMAT_BC7_UNORM_BLOCK, "BC7"},
        {VK_FORMAT_BC7_SRGB_BLOCK, "BC7 sRGB"},  {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8"},
    };
    for (unsigned i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(pd, formats[i].f, &p);
        const VkFormatFeatureFlags o = p.optimalTilingFeatures;
        printf("%-9s sampled %d blit_src %d blit_dst %d transfer_src %d storage %d filter_linear %d\n",
               formats[i].name, !!(o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT), !!(o & VK_FORMAT_FEATURE_BLIT_SRC_BIT),
               !!(o & VK_FORMAT_FEATURE_BLIT_DST_BIT), !!(o & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT),
               !!(o & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT), !!(o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT));
    }
    return 0;
}
