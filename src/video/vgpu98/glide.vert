/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the Glide device's vertices.
 *          Each is 16 floats as the Glide DLL sends them
 *          (vid_vgpu98_glcmd.h, VGPU98_GLV_*): screen x and y, 65535/z,
 *          1/w, the colour (0-255), each TMU's s/w, t/w and 1/w, and the
 *          fog coordinate. Everything is passed on unchanged, without
 *          perspective: the Voodoo iterates its parameters linearly in
 *          screen space and divides per pixel (glide.frag). x and y go to
 *          the Voodoo's 1/16-pixel grid; (0.5, 0.5) is the first pixel's
 *          centre, as in Vulkan; a lower-left origin counts y from the
 *          bottom.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450

layout(std430, set = 0, binding = 1) readonly buffer Verts {
    float v[];
};

layout(push_constant) uniform Draw {
    float width;      /* the session's */
    float height;
    uint  lower_left; /* 1: y counts from the bottom */
    uint  first;      /* the draw's first vertex in the buffer */
} d;

layout(location = 0) noperspective out vec4 v_color; /* r, g, b, a: 0-255 */
layout(location = 1) noperspective out vec4 v_depth; /* 65535/z, 1/w, fog, unused */
layout(location = 2) noperspective out vec3 v_tex0;  /* TMU0's s/w, t/w, 1/w */
layout(location = 3) noperspective out vec3 v_tex1;  /* TMU1's */

void
main()
{
    const uint b = (d.first + uint(gl_VertexIndex)) * 16u;
    float      x = v[b + 0u];
    float      y = v[b + 1u];

    x = floor(x * 16.0 + 0.5) * (1.0 / 16.0);
    y = floor(y * 16.0 + 0.5) * (1.0 / 16.0);
    /* Lower left: the rows flipped, and moved down 1/256 of a pixel, so a
       centre on a horizontal edge falls inside at the edge of lower y in
       the program's rows, as the Voodoo has it (Vulkan's rule takes the
       top edge of the image's). */
    if (d.lower_left != 0u)
        y = d.height - y + (1.0 / 256.0);
    /* Depth comes from the fragment shader: no plane clips it. */
    gl_Position = vec4(2.0 * x / d.width - 1.0, 2.0 * y / d.height - 1.0, 0.5, 1.0);
    v_color     = vec4(v[b + 4u], v[b + 5u], v[b + 6u], v[b + 7u]);
    v_depth     = vec4(v[b + 2u], v[b + 3u], v[b + 14u], 0.0);
    v_tex0      = vec3(v[b + 8u], v[b + 9u], v[b + 10u]);
    v_tex1      = vec3(v[b + 11u], v[b + 12u], v[b + 13u]);
}
