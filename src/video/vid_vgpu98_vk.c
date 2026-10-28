/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the host's Vulkan driver. Loaded at run time,
 *          with a headless Vulkan 1.1 device (no window, no surface):
 *          the GPU renders into images of its own, and the display
 *          composes the screen on the same device.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <86box/plat_dynld.h>
#include <86box/vid_vgpu98_vk.h>

#if defined(_WIN32)
#    define VULKAN_LIBRARY "vulkan-1.dll"
#elif defined(__APPLE__)
#    define VULKAN_LIBRARY "libvulkan.1.dylib"
#else
#    define VULKAN_LIBRARY "libvulkan.so.1"
#endif

typedef void (*vk_log_t)(void *priv, const char *fmt, ...);

int
vgpu98_vk_memory_type(const vgpu98_vk_t *vk, uint32_t bits, VkMemoryPropertyFlags props)
{
    uint32_t i;

    for (i = 0; i < vk->mem.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && ((vk->mem.memoryTypes[i].propertyFlags & props) == props))
            return (int) i;
    }
    return -1;
}

int
vgpu98_vk_format_has(const vgpu98_vk_t *vk, VkFormat format, VkFormatFeatureFlags features)
{
    VkFormatProperties p;

    vk->vkGetPhysicalDeviceFormatProperties(vk->phys, format, &p);
    return (p.optimalTilingFeatures & features) == features;
}

static int
buffer_create(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b, VkDeviceSize size, VkBufferUsageFlags usage, int local)
{
    VkBufferCreateInfo   bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    VkMemoryRequirements req;
    int                  type;

    memset(b, 0, sizeof(*b));
    bi.size        = size;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk->vkCreateBuffer(vk->dev, &bi, NULL, &b->buffer) != VK_SUCCESS)
        return 0;
    vk->vkGetBufferMemoryRequirements(vk->dev, b->buffer, &req);
    if (local) {
        type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0)
            type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, 0);
    } else {
        /* Host-visible and coherent, cached where the host offers it: these
           buffers are read back by the CPU as often as they are written. */
        type = vgpu98_vk_memory_type(vk, req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (type < 0)
            type = vgpu98_vk_memory_type(vk, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (type < 0)
        goto fail;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = (uint32_t) type;
    if (vk->vkAllocateMemory(vk->dev, &ai, NULL, &b->memory) != VK_SUCCESS)
        goto fail;
    if (vk->vkBindBufferMemory(vk->dev, b->buffer, b->memory, 0) != VK_SUCCESS)
        goto fail;
    if (!local && (vk->vkMapMemory(vk->dev, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map) != VK_SUCCESS))
        goto fail;
    b->size = size;
    return 1;

fail:
    vgpu98_vk_buffer_destroy(vk, b);
    return 0;
}

int
vgpu98_vk_buffer_create(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    return buffer_create(vk, b, size, usage, 0);
}

int
vgpu98_vk_buffer_create_local(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    return buffer_create(vk, b, size, usage, 1);
}

VkResult
vgpu98_vk_submit(vgpu98_vk_t *vk, const VkSubmitInfo *si, VkFence fence)
{
    VkResult res;

    thread_wait_mutex(vk->queue_lock);
    res = vk->vkQueueSubmit(vk->queue, 1, si, fence);
    thread_release_mutex(vk->queue_lock);
    return res;
}

VkResult
vgpu98_vk_wait_idle(vgpu98_vk_t *vk)
{
    VkResult res;

    thread_wait_mutex(vk->queue_lock);
    res = vk->vkDeviceWaitIdle(vk->dev);
    thread_release_mutex(vk->queue_lock);
    return res;
}

VkResult
vgpu98_vk_wait_fence(vgpu98_vk_t *vk, VkFence fence, int (*second)(void *priv, uint32_t seconds), void *priv)
{
    uint32_t seconds = 0;

    for (;;) {
        const VkResult res = vk->vkWaitForFences(vk->dev, 1, &fence, VK_TRUE, 1000000000ull);

        if (res != VK_TIMEOUT)
            return res;
        seconds++;
        if (second && second(priv, seconds))
            return VK_TIMEOUT;
    }
}

void
vgpu98_vk_buffer_destroy(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b)
{
    if (b->buffer)
        vk->vkDestroyBuffer(vk->dev, b->buffer, NULL);
    if (b->memory)
        vk->vkFreeMemory(vk->dev, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

/* Prefers a discrete GPU, then an integrated one, then anything with a
   graphics queue and Vulkan 1.1. */
static int
pick_device(vgpu98_vk_t *vk, vk_log_t log, void *priv)
{
    VkPhysicalDevice *devs  = NULL;
    uint32_t          n     = 0;
    int               best  = -1;
    int               score = -1;
    uint32_t          i;

    if ((vk->vkEnumeratePhysicalDevices(vk->instance, &n, NULL) != VK_SUCCESS) || !n)
        return 0;
    devs = calloc(n, sizeof(*devs));
    if (vk->vkEnumeratePhysicalDevices(vk->instance, &n, devs) != VK_SUCCESS) {
        free(devs);
        return 0;
    }
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        VkQueueFamilyProperties   *q;
        uint32_t                   nq = 0;
        uint32_t                   k;
        int                        s;
        int                        graphics = -1;

        vk->vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_1)
            continue;
        vk->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        q = calloc(nq ? nq : 1, sizeof(*q));
        vk->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, q);
        for (k = 0; k < nq; k++) {
            if (q[k].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphics = (int) k;
                break;
            }
        }
        free(q);
        if (graphics < 0)
            continue;
        s = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 3 :
            (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 2 : 1;
        log(priv, "VGPU98 Vulkan: found %s (type %d, API %u.%u)\n", p.deviceName, p.deviceType,
            VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
        if (s > score) {
            score            = s;
            best             = (int) i;
            vk->queue_family = (uint32_t) graphics;
        }
    }
    if (best >= 0)
        vk->phys = devs[best];
    free(devs);
    return best >= 0;
}

static int
has_extension(vgpu98_vk_t *vk, const char *name)
{
    VkExtensionProperties *e = NULL;
    uint32_t               n = 0;
    uint32_t               i;
    int                    found = 0;

    if ((vk->vkEnumerateDeviceExtensionProperties(vk->phys, NULL, &n, NULL) != VK_SUCCESS) || !n)
        return 0;
    e = calloc(n, sizeof(*e));
    if (vk->vkEnumerateDeviceExtensionProperties(vk->phys, NULL, &n, e) == VK_SUCCESS) {
        for (i = 0; i < n; i++) {
            if (strcmp(e[i].extensionName, name) == 0)
                found = 1;
        }
    }
    free(e);
    return found;
}

/* The device, with custom border colours when vk has them (their features
   chained here). */
static VkResult
create_device(vgpu98_vk_t *vk, VkDeviceCreateInfo *dci, VkPhysicalDeviceCustomBorderColorFeaturesEXT *border)
{
    const char *extensions[1] = { VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME };

    border->pNext                = NULL;
    dci->enabledExtensionCount   = vk->border_custom ? 1 : 0;
    dci->ppEnabledExtensionNames = vk->border_custom ? extensions : NULL;
    dci->pNext                   = vk->border_custom ? border : NULL;
    vk->dev                      = VK_NULL_HANDLE;
    if (vk->vkCreateDevice(vk->phys, dci, NULL, &vk->dev) != VK_SUCCESS)
        vk->dev = VK_NULL_HANDLE;
    /* Nothing of this call's stays in the create info. */
    dci->enabledExtensionCount   = 0;
    dci->ppEnabledExtensionNames = NULL;
    dci->pNext                   = NULL;
    return vk->dev ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

vgpu98_vk_t *
vgpu98_vk_create(vk_log_t log, void *priv)
{
    vgpu98_vk_t             *vk  = calloc(1, sizeof(vgpu98_vk_t));
    VkApplicationInfo        app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    VkInstanceCreateInfo     ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkDeviceQueueCreateInfo  qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    VkDeviceCreateInfo       dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    VkPhysicalDeviceFeatures have;
    VkPhysicalDeviceFeatures want;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT
    };
    float       priority = 1.0f;
    dllimp_t    imports[] = {
        { "vkGetInstanceProcAddr", &vk->vkGetInstanceProcAddr },
        { NULL, NULL }
    };

    vk->lib = dynld_module(VULKAN_LIBRARY, imports);
    if (!vk->lib) {
        log(priv, "VGPU98 Vulkan: no Vulkan driver on the host (%s); the GPU has no 3D\n", VULKAN_LIBRARY);
        free(vk);
        return NULL;
    }
    vk->vkCreateInstance = (PFN_vkCreateInstance) vk->vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (!vk->vkCreateInstance)
        goto fail;

    app.pApplicationName   = "86Box VGPU98";
    app.applicationVersion = 1;
    app.pEngineName        = "VGPU98";
    app.engineVersion      = 1;
    app.apiVersion         = VK_API_VERSION_1_1;
    ici.pApplicationInfo   = &app;
    if (vk->vkCreateInstance(&ici, NULL, &vk->instance) != VK_SUCCESS) {
        log(priv, "VGPU98 Vulkan: vkCreateInstance failed\n");
        goto fail;
    }
#define VGPU98_VK_LOAD_INSTANCE(name)                                                       \
    vk->name = (PFN_##name) vk->vkGetInstanceProcAddr(vk->instance, #name);                 \
    if (!vk->name) {                                                                        \
        log(priv, "VGPU98 Vulkan: %s missing\n", #name);                                    \
        goto fail;                                                                          \
    }
    VGPU98_VK_INSTANCE_FUNCS(VGPU98_VK_LOAD_INSTANCE)
#undef VGPU98_VK_LOAD_INSTANCE

    if (!pick_device(vk, log, priv)) {
        log(priv, "VGPU98 Vulkan: no device with Vulkan 1.1 and a graphics queue\n");
        goto fail;
    }
    vk->vkGetPhysicalDeviceProperties(vk->phys, &vk->props);
    vk->vkGetPhysicalDeviceMemoryProperties(vk->phys, &vk->mem);
    vk->vkGetPhysicalDeviceFeatures(vk->phys, &have);
    vk->vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2) vk->vkGetInstanceProcAddr(vk->instance,
                                                                                                    "vkGetPhysicalDeviceFeatures2");

    memset(&want, 0, sizeof(want));
    want.samplerAnisotropy = have.samplerAnisotropy;
    want.depthClamp        = have.depthClamp;
    /* The vertex pipeline (spec/gpu-commands.md): user clip planes as clip
       distances, and the line rule's integer arithmetic. */
    want.shaderClipDistance = have.shaderClipDistance;
    want.shaderInt64        = have.shaderInt64;
    /* The fragment shader's texture arrays, indexed by its stage loop. */
    want.shaderSampledImageArrayDynamicIndexing = have.shaderSampledImageArrayDynamicIndexing;
    /* Draws straight from resident index buffers hand Vulkan the guest's
       32-bit indices. */
    want.fullDrawIndexUint32 = have.fullDrawIndexUint32;
    /* The OpenGL device's logic op. */
    want.logicOp = have.logicOp;
    vk->anisotropy          = have.samplerAnisotropy;
    vk->depth_clamp         = have.depthClamp;
    vk->clip_distance       = have.shaderClipDistance;
    vk->int64               = have.shaderInt64;
    vk->sampler_indexing    = have.shaderSampledImageArrayDynamicIndexing;
    vk->full_index32        = have.fullDrawIndexUint32;
    vk->logic_op            = have.logicOp;

    qci.queueFamilyIndex = vk->queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    dci.pEnabledFeatures     = &want;
    /* D3D's BORDERCOLOR is any colour; core Vulkan offers three: custom
       border colours, without a format, when the device reports them
       (without a features query, tried as ever). */
    if (has_extension(vk, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME)) {
        vk->border_custom = 1;
        if (vk->vkGetPhysicalDeviceFeatures2) {
            VkPhysicalDeviceFeatures2                    f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            VkPhysicalDeviceCustomBorderColorFeaturesEXT q  = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT };

            f2.pNext = &q;
            vk->vkGetPhysicalDeviceFeatures2(vk->phys, &f2);
            vk->border_custom = q.customBorderColors && q.customBorderColorWithoutFormat;
        }
    }
    if (vk->border_custom) {
        border.customBorderColors             = VK_TRUE;
        border.customBorderColorWithoutFormat = VK_TRUE;
    }
    if (create_device(vk, &dci, &border) != VK_SUCCESS) {
        /* A device that refuses the extension after all: without it. */
        if (vk->border_custom) {
            vk->border_custom = 0;
            create_device(vk, &dci, &border);
        }
        if (!vk->dev) {
            log(priv, "VGPU98 Vulkan: vkCreateDevice failed\n");
            goto fail;
        }
    }
#define VGPU98_VK_LOAD_DEVICE(name)                                                     \
    vk->name = (PFN_##name) vk->vkGetDeviceProcAddr(vk->dev, #name);                    \
    if (!vk->name) {                                                                    \
        log(priv, "VGPU98 Vulkan: %s missing\n", #name);                                \
        goto fail;                                                                      \
    }
    VGPU98_VK_DEVICE_FUNCS(VGPU98_VK_LOAD_DEVICE)
#undef VGPU98_VK_LOAD_DEVICE
    vk->vkGetDeviceQueue(vk->dev, vk->queue_family, 0, &vk->queue);
    vk->queue_lock = thread_create_mutex();

    log(priv,
        "VGPU98 Vulkan: using %s, driver %08X; anisotropy %s, custom border colours %s, clip distances %s, "
        "64-bit shader integers %s, indexed texture arrays %s\n",
        vk->props.deviceName, vk->props.driverVersion, vk->anisotropy ? "yes" : "no",
        vk->border_custom ? "yes" : "no", vk->clip_distance ? "yes" : "no", vk->int64 ? "yes" : "no",
        vk->sampler_indexing ? "yes" : "no");
    return vk;

fail:
    vgpu98_vk_destroy(vk);
    return NULL;
}

void
vgpu98_vk_destroy(vgpu98_vk_t *vk)
{
    if (!vk)
        return;
    if (vk->dev && vk->vkDestroyDevice) {
        vk->vkDeviceWaitIdle(vk->dev);
        vk->vkDestroyDevice(vk->dev, NULL);
    }
    if (vk->instance && vk->vkDestroyInstance)
        vk->vkDestroyInstance(vk->instance, NULL);
    if (vk->lib)
        dynld_close(vk->lib);
    if (vk->queue_lock)
        thread_close_mutex(vk->queue_lock);
    free(vk);
}
