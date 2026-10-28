/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the host's Vulkan driver, loaded at run time
 *          (vid_vgpu98_vk.c), for the 3D renderer (vid_vgpu98_r3d.c) and
 *          the display (vid_vgpu98_disp.c), which share one device and
 *          one queue. 86Box links no Vulkan loader: if the host has none,
 *          the GPU simply has no 3D.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#ifndef VIDEO_VGPU98_VK_H
#define VIDEO_VGPU98_VK_H

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <86box/thread.h>

#define VGPU98_VK_INSTANCE_FUNCS(X)              \
    X(vkDestroyInstance)                         \
    X(vkEnumeratePhysicalDevices)                \
    X(vkGetPhysicalDeviceProperties)             \
    X(vkGetPhysicalDeviceFeatures)               \
    X(vkGetPhysicalDeviceQueueFamilyProperties)  \
    X(vkGetPhysicalDeviceMemoryProperties)       \
    X(vkGetPhysicalDeviceFormatProperties)       \
    X(vkGetPhysicalDeviceImageFormatProperties)  \
    X(vkEnumerateDeviceExtensionProperties)      \
    X(vkCreateDevice)                            \
    X(vkGetDeviceProcAddr)

#define VGPU98_VK_DEVICE_FUNCS(X)  \
    X(vkDestroyDevice)             \
    X(vkGetDeviceQueue)            \
    X(vkQueueSubmit)               \
    X(vkDeviceWaitIdle)            \
    X(vkAllocateMemory)            \
    X(vkFreeMemory)                \
    X(vkMapMemory)                 \
    X(vkUnmapMemory)               \
    X(vkCreateBuffer)              \
    X(vkDestroyBuffer)             \
    X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory)          \
    X(vkCreateImage)               \
    X(vkDestroyImage)              \
    X(vkGetImageMemoryRequirements) \
    X(vkBindImageMemory)           \
    X(vkCreateImageView)           \
    X(vkDestroyImageView)          \
    X(vkCreateSampler)             \
    X(vkDestroySampler)            \
    X(vkCreateShaderModule)        \
    X(vkDestroyShaderModule)       \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout)      \
    X(vkDestroyPipelineLayout)     \
    X(vkCreateDescriptorPool)      \
    X(vkDestroyDescriptorPool)     \
    X(vkResetDescriptorPool)       \
    X(vkAllocateDescriptorSets)    \
    X(vkUpdateDescriptorSets)      \
    X(vkCreateRenderPass)          \
    X(vkDestroyRenderPass)         \
    X(vkCreateFramebuffer)         \
    X(vkDestroyFramebuffer)        \
    X(vkCreateGraphicsPipelines)   \
    X(vkCreatePipelineCache)       \
    X(vkDestroyPipelineCache)      \
    X(vkGetPipelineCacheData)      \
    X(vkMergePipelineCaches)       \
    X(vkDestroyPipeline)           \
    X(vkCreateCommandPool)         \
    X(vkDestroyCommandPool)        \
    X(vkAllocateCommandBuffers)    \
    X(vkFreeCommandBuffers)        \
    X(vkBeginCommandBuffer)        \
    X(vkEndCommandBuffer)          \
    X(vkResetCommandBuffer)        \
    X(vkCreateFence)               \
    X(vkDestroyFence)              \
    X(vkWaitForFences)             \
    X(vkGetFenceStatus)            \
    X(vkResetFences)               \
    X(vkCmdPipelineBarrier)        \
    X(vkCmdCopyBufferToImage)      \
    X(vkCmdCopyImageToBuffer)      \
    X(vkCmdCopyImage)              \
    X(vkCmdBeginRenderPass)        \
    X(vkCmdEndRenderPass)          \
    X(vkCmdBindPipeline)           \
    X(vkCmdBindDescriptorSets)     \
    X(vkCmdBindIndexBuffer)        \
    X(vkCmdDraw)                   \
    X(vkCmdDrawIndexed)            \
    X(vkCmdSetViewport)            \
    X(vkCmdSetScissor)             \
    X(vkCmdSetStencilCompareMask)  \
    X(vkCmdSetStencilWriteMask)    \
    X(vkCmdSetStencilReference)    \
    X(vkCmdSetDepthBias)           \
    X(vkCmdPushConstants)          \
    X(vkCmdClearAttachments)       \
    X(vkCmdClearColorImage)        \
    X(vkCmdClearDepthStencilImage) \
    X(vkCreateComputePipelines)    \
    X(vkCmdDispatch)               \
    X(vkCmdCopyBuffer)

typedef struct vgpu98_vk_t {
    void                     *lib;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkCreateInstance      vkCreateInstance;
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2; /* Vulkan 1.1's; NULL if missing */
#define VGPU98_VK_DECLARE(name) PFN_##name name;
    VGPU98_VK_INSTANCE_FUNCS(VGPU98_VK_DECLARE)
    VGPU98_VK_DEVICE_FUNCS(VGPU98_VK_DECLARE)
#undef VGPU98_VK_DECLARE

    VkInstance                       instance;
    VkPhysicalDevice                 phys;
    VkDevice                         dev;
    VkQueue                          queue;
    uint32_t                         queue_family;
    VkPhysicalDeviceProperties       props;
    VkPhysicalDeviceMemoryProperties mem;
    int                              anisotropy;    /* samplerAnisotropy enabled */
    int                              depth_clamp;   /* depthClamp enabled */
    int                              border_custom; /* VK_EXT_custom_border_color enabled */
    int                              clip_distance; /* shaderClipDistance enabled */
    int                              int64;         /* shaderInt64 enabled */
    int                              sampler_indexing; /* shaderSampledImageArrayDynamicIndexing enabled */
    int                              full_index32;  /* fullDrawIndexUint32 enabled: any 32-bit index value */
    int                              logic_op;      /* logicOp enabled (the OpenGL device's) */
    mutex_t                         *queue_lock;    /* the queue is used by two threads */
} vgpu98_vk_t;

/* Loads the host's Vulkan driver and makes a headless device with a graphics
   queue. NULL if there is none (logged through `log`). */
vgpu98_vk_t *vgpu98_vk_create(void (*log)(void *priv, const char *fmt, ...), void *priv);
void         vgpu98_vk_destroy(vgpu98_vk_t *vk);

/* A memory type with the properties among `bits`; -1 if none. */
int      vgpu98_vk_memory_type(const vgpu98_vk_t *vk, uint32_t bits, VkMemoryPropertyFlags props);
/* Does the format have these optimal-tiling features? */
int      vgpu98_vk_format_has(const vgpu98_vk_t *vk, VkFormat format, VkFormatFeatureFlags features);

/* A buffer with its own memory: host-visible ones are mapped. */
typedef struct vgpu98_vk_buffer_t {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    void          *map;
    VkDeviceSize   size;
} vgpu98_vk_buffer_t;

/* A host-visible, mapped buffer. */
int  vgpu98_vk_buffer_create(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b, VkDeviceSize size, VkBufferUsageFlags usage);
/* A buffer in the host GPU's own memory, not mapped. */
int  vgpu98_vk_buffer_create_local(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b, VkDeviceSize size, VkBufferUsageFlags usage);
void vgpu98_vk_buffer_destroy(vgpu98_vk_t *vk, vgpu98_vk_buffer_t *b);

/* vkQueueSubmit and vkDeviceWaitIdle under the queue lock: the 3D renderer
   and the display submit from their own threads. */
VkResult vgpu98_vk_submit(vgpu98_vk_t *vk, const VkSubmitInfo *si, VkFence fence);
VkResult vgpu98_vk_wait_idle(vgpu98_vk_t *vk);
/* Waits for a fence a second at a time, never without a limit: after each
   second its work is not done, `second` (if any) is told how many so far,
   and its nonzero return gives the wait up (VK_TIMEOUT). Otherwise as
   vkWaitForFences: VK_SUCCESS, or the failure (a device lost). */
VkResult vgpu98_vk_wait_fence(vgpu98_vk_t *vk, VkFence fence, int (*second)(void *priv, uint32_t seconds), void *priv);

/* The 3D renderer's device, for the display (vid_vgpu98_r3d.c). */
struct vgpu98_r3d_t;
vgpu98_vk_t *vgpu98_r3d_vk(struct vgpu98_r3d_t *r);

/* The Glide device's use of the renderer (vid_vgpu98_glide.c),
   on the command processor's thread.
   It records into the renderer's command buffers: `end`, given once, ends
   its render pass, and the renderer calls it before recording for itself
   and before every submission (its pass may thus end under it: the device
   begins it again when it next draws). */
void            vgpu98_r3d_glide_hook(struct vgpu98_r3d_t *r, void (*end)(void *priv), void *priv);
/* The buffer to record into, a slot begun if need be; the renderer's own
   render pass ended and its pipeline forgotten. */
VkCommandBuffer vgpu98_r3d_glide_cmd(struct vgpu98_r3d_t *r);
/* The buffer being recorded, nothing begun or ended (for `end`). */
VkCommandBuffer vgpu98_r3d_glide_rec(struct vgpu98_r3d_t *r);
/* Room in the current slot's host-visible buffers, `which` 0 staging
   (transfers), 1 storage, 2 uniform; NULL with 3D stopped. A full slot is
   submitted first, which ends the Glide device's pass. */
void           *vgpu98_r3d_glide_alloc(struct vgpu98_r3d_t *r, int which, VkDeviceSize size, VkDeviceSize align,
                                       VkBuffer *buf, VkDeviceSize *off);
/* The slot being recorded (one begun if need be) and its generation, which
   changes each time the slot is begun again: the device's descriptor pool
   for the slot is reset then. */
uint32_t        vgpu98_r3d_glide_slot(struct vgpu98_r3d_t *r, uint32_t *gen);
#define VGPU98_R3D_SLOTS 3
/* An object destroyed once the slot's work is done: 0 an image, 1 a view,
   2 memory, 3 a framebuffer, 4 a vgpu98_vk_buffer_t (malloc'd). */
void            vgpu98_r3d_glide_dead(struct vgpu98_r3d_t *r, int kind, uint64_t handle);
/* Submits what was recorded; with `wait`, until the host GPU has done it.
   0 if it failed (3D stops). */
int             vgpu98_r3d_glide_submit(struct vgpu98_r3d_t *r, int wait);
int             vgpu98_r3d_glide_image_memory(struct vgpu98_r3d_t *r, VkImage image, VkDeviceMemory *mem, VkDeviceSize *bytes);
/* The pipeline cache, kept on disk with the renderer's pipelines; a new
   pipeline made in it is told, so it is saved in time. */
VkPipelineCache vgpu98_r3d_glide_pcache(struct vgpu98_r3d_t *r);
void            vgpu98_r3d_glide_new_pipeline(struct vgpu98_r3d_t *r);
/* 3D stopped (a failure); a failure of the Glide device's own stops it. */
int             vgpu98_r3d_broken(const struct vgpu98_r3d_t *r);
void            vgpu98_r3d_glide_fail(struct vgpu98_r3d_t *r, const char *what, int res);

/* The OpenGL device's use of the renderer (vid_vgpu98_ogl.c):
   a hook and a command buffer of its own,
   as the Glide device's (taking the buffer ends the Glide device's render
   pass, as Glide's ends this one's); everything else through the functions
   above. */
void            vgpu98_r3d_ogl_hook(struct vgpu98_r3d_t *r, void (*end)(void *priv), void *priv);
VkCommandBuffer vgpu98_r3d_ogl_cmd(struct vgpu98_r3d_t *r);

#endif /* VIDEO_VGPU98_VK_H */
