/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the OpenGL device's fragments:
 *          the colour (smooth or
 *          flat, the face's with two-sided lighting), texturing as GL 1.1
 *          defines it, fog (linear, exp, exp2, per fragment), smooth
 *          points' and lines' coverage in alpha, polygon and line stipple,
 *          the user's clip planes and the alpha test in 8-bit fixed point.
 *          Scissor, stencil, depth, blending, the logic op and the masks
 *          are Vulkan's (the pipeline).
 *
 *          Texturing: the texels are fetched from the device's texel arena
 *          (RGBA8, each level with its border) and filtered here, GL's
 *          way: lambda from the coordinates' derivatives, the switch
 *          between magnification and minification with its c, REPEAT and
 *          CLAMP, border texels and the border colour, the six filters,
 *          the mipmaps; then the texture environment by base format.
 *          Command set 2.6: two
 *          texture units, unit 1 sampled and applied on unit 0's result;
 *          index textures, each texel looked up in a palette. Command set
 *          2.7: clamp-to-edge, never sampling
 *          the border.
 *
 *          Pixel rectangles (PIXELS): each fragment's pixel of the image
 *          (zoomed), its colour, depth, stencil bit or bitmap bit, then the
 *          same path as other fragments, with the raster position's
 *          texture coordinates and fog distance.
 *
 *          COMPUTED_DEPTH: depth on the 24-bit grid, for a host whose
 *          depth and stencil format is D32F-S8, and for
 *          depth images; polygon offset then applied here.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450

#define R_ENABLES        60u
#define R_CLIPS_ON       62u
#define R_SHADE          63u
#define R_FOG_MODE       361u
#define R_FOG_DENSITY    362u
#define R_FOG_START      363u
#define R_FOG_END        364u
#define R_FOG_COLOR      365u
#define R_TEX_ENV_MODE   370u
#define R_TEX_ENV_COLOR  371u
#define R_ALPHA_FUNC     375u
#define R_ALPHA_REF      376u
#define R_LINE_STIPPLE   395u
#define R_POLYGON_OFFSET 396u
#define R_POLYGON_STIPPLE 401u
#define R_UNIT1          433u /* 2.6: texture unit 1's registers (VGPU98_OGL_U_*): */
#define U_TEX_ENV_MODE   50u
#define U_TEX_ENV_COLOR  51u
#define R_VEC4S          124u /* the registers' uvec4s (VGPU98_OGL_R_COUNT, rounded up) */
#define R_TEX_BLOCK      R_VEC4S /* in uvec4s: each unit's texture drawn with, 16 each */

#define EN_FOG           0x00000008u
#define EN_ALPHA_TEST    0x00000010u
#define EN_OFFSET_FILL   0x00000800u
#define EN_OFFSET_LINE   0x00001000u
#define EN_OFFSET_POINT  0x00002000u
#define EN_POLY_STIPPLE  0x00004000u
#define EN_LINE_STIPPLE  0x00008000u

#define KIND_POINTS    0u
#define KIND_LINES     1u
#define KIND_TRIANGLES 2u
#define KIND_PIXELS    14u
#define KIND_CLEAR     15u
#define DRAW_CLIP      0x100u
#define DRAW_POLYGON   0x400u

#define PIXELS_COLOR   0u
#define PIXELS_DEPTH   1u
#define PIXELS_STENCIL 2u
#define PIXELS_BITMAP  3u

#define BASE_ALPHA     1u
#define BASE_LUMINANCE 2u
#define BASE_LUM_ALPHA 3u
#define BASE_INTENSITY 4u
#define BASE_RGB       5u

#define TEX_INDEXED    0x01000000u /* the block's [0]: an index texture */

#define ENV_MODULATE   0u
#define ENV_DECAL      1u
#define ENV_BLEND      2u
#define ENV_REPLACE    3u

/* The registers, then each unit's texture drawn with (16 uvec4s each):
   - [0] base format | dimensions << 4 | wrap s << 8 | wrap t << 12 (0
     REPEAT, 1 CLAMP, 2 clamp-to-edge) |
     minification filter << 16 | magnification filter << 20 |
     TEX_INDEXED (0: none); its levels; its border | the palette's width
     less one << 8; the palette's first entry in the arena;
   - [1] the border colour (floats);
   - [2 + k] level k: its first texel in the arena (an index level's first
     dword), its width and height (the border included).
   An index texture (2.6) has a byte a texel, rows not padded, looked up in
   its palette (RGBA8) before filtering; the base format is the palette's. */
layout(std140, set = 0, binding = 0) uniform Regs {
    uvec4 r4[R_VEC4S + 32u];
};
/* The vertices, or a PIXELS image */
layout(std430, set = 0, binding = 1) readonly buffer Verts {
    uint data[];
};
/* The texel arena: RGBA8, red in the low byte */
layout(std430, set = 0, binding = 3) readonly buffer Arena {
    uint texels[];
};

layout(push_constant) uniform Draw {
    uint control;
    uint height;
    uint width;
    uint bit;
    vec4 pos;
    vec4 zoom;
    vec4 color;
    vec4 tex;
    vec4 tex1;
} d;

layout(location = 0) in vec4 i_front;
layout(location = 1) in vec4 i_back;
layout(location = 2) flat in vec4 i_front_flat;
layout(location = 3) flat in vec4 i_back_flat;
layout(location = 4) in vec4 i_tex;
layout(location = 5) in float i_fog;
layout(location = 6) in vec4 i_clip0;
layout(location = 7) in vec2 i_clip1;
layout(location = 8) flat in vec4 i_shape;
layout(location = 9) flat in vec4 i_shape2;
layout(location = 10) in vec4 i_tex1;

layout(location = 0) out vec4 o_color;

uint
R(uint i)
{
    return r4[i >> 2][i & 3u];
}

float
RF(uint i)
{
    return uintBitsToFloat(R(i));
}

uvec4
T(uint u, uint i)
{
    return r4[R_TEX_BLOCK + 16u * u + i];
}

/* A compare function (VGPU98_OGL_CMP_*) of a against b */
bool
compare(uint f, uint a, uint b)
{
    switch (f) {
        case 0u:
            return false;
        case 1u:
            return a < b;
        case 2u:
            return a == b;
        case 3u:
            return a <= b;
        case 4u:
            return a > b;
        case 5u:
            return a != b;
        case 6u:
            return a >= b;
        default:
            return true;
    }
}

/* --- Texturing (GL 1.1, 3.8) --------------------------------------------- */

int
imod(int i, int n)
{
    const int m = i % n;

    return (m < 0) ? m + n : m;
}

/* Texel (i, j) of a level (j 0 for 1D), indices from GL's (the border at
   -1): the border colour outside the image */
vec4
fetch(uint u, uvec4 lv, int i, int j, int border, bool one_d)
{
    const int x = i + border;
    const int y = one_d ? 0 : j + border;

    if ((x < 0) || (x >= int(lv.y)) || (y < 0) || (y >= int(lv.z)))
        return uintBitsToFloat(T(u, 1u));
    if ((T(u, 0u).x & TEX_INDEXED) != 0u) {
        /* An index, a byte each, within the palette's width; its entry
           (2.6) */
        const uvec4 h = T(u, 0u);
        const uint  n = uint(y) * lv.y + uint(x);
        const uint  i = (texels[lv.x + (n >> 2)] >> ((n & 3u) * 8u)) & (h.z >> 8);

        return unpackUnorm4x8(texels[h.w + i]);
    }
    return unpackUnorm4x8(texels[lv.x + uint(y) * lv.y + uint(x)]);
}

/* A level sampled at (s, t) by NEAREST or LINEAR */
vec4
sample_level(uint tu, uint k, vec2 st, bool linear)
{
    const uvec4 h     = T(tu, 0u);
    const uvec4 lv    = T(tu, 2u + k);
    const bool  one_d = ((h.x >> 4) & 15u) == 1u;
    const bool  clamp_s = ((h.x >> 8) & 15u) != 0u;
    const bool  clamp_t = ((h.x >> 12) & 15u) != 0u;
    /* 2.7: clamp-to-edge (2), the border never sampled */
    const bool  edge_s  = ((h.x >> 8) & 15u) == 2u;
    const bool  edge_t  = ((h.x >> 12) & 15u) == 2u;
    const int   b     = int(h.z & 255u);
    const int   wn    = max(int(lv.y) - 2 * b, 1);
    const int   hn    = one_d ? 1 : max(int(lv.z) - 2 * b, 1);
    float       s     = clamp_s ? clamp(st.x, 0.0, 1.0) : st.x;
    float       t     = clamp_t ? clamp(st.y, 0.0, 1.0) : st.y;
    const float u     = s * float(wn);
    const float v     = t * float(hn);

    if (!linear) {
        /* i = floor(u), REPEAT's mod; CLAMP's s of 1 gives the last */
        int i = int(floor(u)), j = int(floor(v));

        i = clamp_s ? min(i, wn - 1) : imod(i, wn);
        j = one_d ? 0 : (clamp_t ? min(j, hn - 1) : imod(j, hn));
        return fetch(tu, lv, i, j, b, one_d);
    }
    {
        /* The four texels around (u - 1/2, v - 1/2); CLAMP reaches the
           border there, clamp-to-edge keeps both inside the level */
        const float uu = u - 0.5, vv = v - 0.5;
        int         i0 = int(floor(uu)), j0 = int(floor(vv));
        int         i1 = i0 + 1, j1 = j0 + 1;
        const float a  = fract(uu), c = fract(vv);

        if (edge_s) {
            i0 = clamp(i0, 0, wn - 1);
            i1 = clamp(i1, 0, wn - 1);
        } else if (!clamp_s) {
            i0 = imod(i0, wn);
            i1 = imod(i1, wn);
        }
        if (one_d)
            return mix(fetch(tu, lv, i0, 0, b, true), fetch(tu, lv, i1, 0, b, true), a);
        if (edge_t) {
            j0 = clamp(j0, 0, hn - 1);
            j1 = clamp(j1, 0, hn - 1);
        } else if (!clamp_t) {
            j0 = imod(j0, hn);
            j1 = imod(j1, hn);
        }
        return mix(mix(fetch(tu, lv, i0, j0, b, false), fetch(tu, lv, i1, j0, b, false), a),
                   mix(fetch(tu, lv, i0, j1, b, false), fetch(tu, lv, i1, j1, b, false), a), c);
    }
}

/* The texture's value at (s, t), its lambda from the derivatives given */
vec4
sample_texture(uint u, vec2 st, vec2 dx, vec2 dy)
{
    const uvec4 h      = T(u, 0u);
    const uvec4 lv0    = T(u, 2u);
    const bool  one_d  = ((h.x >> 4) & 15u) == 1u;
    const uint  minf   = (h.x >> 16) & 15u;
    const uint  magf   = (h.x >> 20) & 15u;
    const int   q      = int(h.y) - 1;
    const int   b      = int(h.z & 255u);
    const float wn     = float(max(int(lv0.y) - 2 * b, 1));
    const float hn     = one_d ? 0.0 : float(max(int(lv0.z) - 2 * b, 1));
    const float rho    = max(length(vec2(dx.x * wn, dx.y * hn)), length(vec2(dy.x * wn, dy.y * hn)));
    const float lambda = log2(rho);
    /* c: 0.5 for LINEAR magnification with NEAREST_MIPMAP_NEAREST or
       NEAREST_MIPMAP_LINEAR minification (GL 1.1, 3.8.6; audit 4, item 1) */
    const float c      = ((magf == 1u) && ((minf == 2u) || (minf == 4u))) ? 0.5 : 0.0;

    if (!(lambda > c))
        return sample_level(u, 0u, st, magf == 1u);
    switch (minf) {
        case 0u:
        case 1u:
            return sample_level(u, 0u, st, minf == 1u);
        case 2u:
        case 3u: {
            /* The nearest mipmap (GL 1.1, 3.8.7) */
            const int dlev = (lambda <= 0.5) ? 0 : min(int(ceil(lambda + 0.5)) - 1, q);

            return sample_level(u, uint(dlev), st, minf == 3u);
        }
        default: {
            /* Between the two mipmaps around lambda */
            if (lambda >= float(q))
                return sample_level(u, uint(q), st, minf == 5u);
            {
                const int d1 = int(floor(lambda));

                return mix(sample_level(u, uint(d1), st, minf == 5u), sample_level(u, uint(d1 + 1), st, minf == 5u), fract(lambda));
            }
        }
    }
}

/* The texture environment (GL 1.1, tables 3.17 and 3.18) by base format */
vec4
environment(uint u, vec4 cf, vec4 ct)
{
    const uint base = T(u, 0u).x & 15u;
    const uint mode = R((u != 0u) ? R_UNIT1 + U_TEX_ENV_MODE : R_TEX_ENV_MODE);
    const uint ccr  = (u != 0u) ? R_UNIT1 + U_TEX_ENV_COLOR : R_TEX_ENV_COLOR;
    const vec4 cc   = vec4(RF(ccr), RF(ccr + 1u), RF(ccr + 2u), RF(ccr + 3u));
    const float lt  = ct.r; /* luminance and intensity are kept in red */

    switch (mode) {
        case ENV_REPLACE:
            switch (base) {
                case BASE_ALPHA:
                    return vec4(cf.rgb, ct.a);
                case BASE_LUMINANCE:
                    return vec4(vec3(lt), cf.a);
                case BASE_LUM_ALPHA:
                    return vec4(vec3(lt), ct.a);
                case BASE_INTENSITY:
                    return vec4(lt);
                case BASE_RGB:
                    return vec4(ct.rgb, cf.a);
                default:
                    return ct;
            }
        case ENV_DECAL:
            /* Defined for RGB and RGBA alone */
            if (base == BASE_RGB)
                return vec4(ct.rgb, cf.a);
            if (base > BASE_RGB)
                return vec4(mix(cf.rgb, ct.rgb, ct.a), cf.a);
            return cf;
        case ENV_BLEND:
            switch (base) {
                case BASE_ALPHA:
                    return vec4(cf.rgb, cf.a * ct.a);
                case BASE_LUMINANCE:
                    return vec4(mix(cf.rgb, cc.rgb, lt), cf.a);
                case BASE_LUM_ALPHA:
                    return vec4(mix(cf.rgb, cc.rgb, lt), cf.a * ct.a);
                case BASE_INTENSITY:
                    return mix(cf, cc, lt);
                case BASE_RGB:
                    return vec4(mix(cf.rgb, cc.rgb, ct.rgb), cf.a);
                default:
                    return vec4(mix(cf.rgb, cc.rgb, ct.rgb), cf.a * ct.a);
            }
        default: /* MODULATE */
            switch (base) {
                case BASE_ALPHA:
                    return vec4(cf.rgb, cf.a * ct.a);
                case BASE_LUMINANCE:
                    return vec4(cf.rgb * lt, cf.a);
                case BASE_LUM_ALPHA:
                    return vec4(cf.rgb * lt, cf.a * ct.a);
                case BASE_INTENSITY:
                    return cf * lt;
                case BASE_RGB:
                    return vec4(cf.rgb * ct.rgb, cf.a);
                default:
                    return cf * ct;
            }
    }
}

#ifdef COMPUTED_DEPTH
/* Depth on the 24-bit grid, the polygon offset added first where on (m:
   the depth's largest slope). */
void
put_depth(float z, bool offset, float m)
{
    if (offset)
        z += RF(R_POLYGON_OFFSET) * m + RF(R_POLYGON_OFFSET + 1u) * (1.0 / 16777216.0);
    gl_FragDepth = floor(clamp(z, 0.0, 1.0) * 16777215.0 + 0.5) / 16777215.0;
}
#endif

void
main()
{
    const uint kind   = d.control & 15u;
    const uint pkind  = (d.control >> 4) & 15u;
    const uint en     = R(R_ENABLES);
    /* GL's window coordinates: x as Vulkan's, y counted up from the bottom */
    const vec2 win    = vec2(gl_FragCoord.x, float(d.height) - gl_FragCoord.y);
    /* The texture coordinates and their derivatives, taken before any
       fragment of the quad is discarded */
    const vec4 tc     = (kind == KIND_PIXELS) ? d.tex : i_tex;
    const vec2 st     = tc.xy / ((tc.w != 0.0) ? tc.w : 1.0);
    const vec2 st_dx  = dFdx(st);
    const vec2 st_dy  = dFdy(st);
    const vec4 tc1    = (kind == KIND_PIXELS) ? d.tex1 : i_tex1;
    const vec2 st1    = tc1.xy / ((tc1.w != 0.0) ? tc1.w : 1.0);
    const vec2 st1_dx = dFdx(st1);
    const vec2 st1_dy = dFdy(st1);
    vec4       c;
    float      fog_z  = i_fog;
#ifdef COMPUTED_DEPTH
    const float slope = max(abs(dFdx(gl_FragCoord.z)), abs(dFdy(gl_FragCoord.z)));
    float       z     = gl_FragCoord.z;
#endif

    if (kind == KIND_CLEAR) {
        o_color = d.color;
#ifdef COMPUTED_DEPTH
        put_depth(d.pos.z, false, 0.0);
#endif
        return;
    }
    if (kind == KIND_PIXELS) {
        /* The fragment's pixel of the image (GL 1.1, 3.6.4: the zoomed
           rectangle holding its centre); a bitmap's from its floor */
        const ivec2 size = ivec2(d.zoom.zw);
        ivec2       p;
        uint        rw;

        if (pkind == PIXELS_BITMAP)
            p = ivec2(floor(win)) - ivec2(floor(d.pos.xy));
        else
            p = ivec2(floor((win - d.pos.xy) / d.zoom.xy));
        if ((p.x < 0) || (p.y < 0) || (p.x >= size.x) || (p.y >= size.y))
            discard;
        rw = (pkind >= PIXELS_STENCIL) ? (uint(size.x) + 3u) / 4u : uint(size.x);
        fog_z = d.pos.w;
        if (pkind >= PIXELS_STENCIL) {
            const uint at = uint(p.y) * rw * 4u + uint(p.x);
            const uint v  = (data[at >> 2] >> (8u * (at & 3u))) & 0xffu;

            if (pkind == PIXELS_STENCIL) {
                /* One bit of the stencil indices a draw: the fragments of
                   those whose bit is the value written (ogl_vgpu98_ogl.c) */
                if (((v >> (d.bit & 7u)) & 1u) != (d.bit >> 8))
                    discard;
                o_color = vec4(0.0);
#ifdef COMPUTED_DEPTH
                put_depth(z, false, 0.0);
#endif
                return;
            }
            if (v == 0u)
                discard;
            c = d.color;
        } else if (pkind == PIXELS_DEPTH) {
            c = d.color;
#ifdef COMPUTED_DEPTH
            z = clamp(uintBitsToFloat(data[uint(p.y) * rw + uint(p.x)]), 0.0, 1.0);
#endif
        } else
            c = unpackUnorm4x8(data[uint(p.y) * rw + uint(p.x)]);
    } else {
        /* The user's clip planes (triangles of the device's path: points
           and lines were clipped in ogl.vert) */
        if ((kind == KIND_TRIANGLES) && ((d.control & DRAW_CLIP) == 0u)) {
            const uint clips = R(R_CLIPS_ON);

            if ((((clips & 1u) != 0u) && (i_clip0.x < 0.0)) || (((clips & 2u) != 0u) && (i_clip0.y < 0.0)) ||
                (((clips & 4u) != 0u) && (i_clip0.z < 0.0)) || (((clips & 8u) != 0u) && (i_clip0.w < 0.0)) ||
                (((clips & 16u) != 0u) && (i_clip1.x < 0.0)) || (((clips & 32u) != 0u) && (i_clip1.y < 0.0)))
                discard;
        }
        /* Polygon stipple: the pattern's bit at the window's x and y, mod 32 */
        if ((kind == KIND_TRIANGLES) && ((en & EN_POLY_STIPPLE) != 0u)) {
            const uint x   = uint(win.x) & 31u;
            const uint row = R(R_POLYGON_STIPPLE + (uint(win.y) & 31u));

            if (((row >> (31u - x)) & 1u) == 0u)
                discard;
        }
        /* The colour: smooth or flat; the back face's for a back-facing
           triangle (the same as the front's without two-sided lighting) */
        {
            const bool flat_shade = R(R_SHADE) == 0u;
            const bool back       = (kind == KIND_TRIANGLES) && !gl_FrontFacing;

            if (flat_shade)
                c = back ? i_back_flat : i_front_flat;
            else
                c = back ? i_back : i_front;
        }
        /* Line stipple: the counter at this fragment, along the major axis
           from the line's start */
        if ((kind == KIND_LINES) && ((en & EN_LINE_STIPPLE) != 0u)) {
            const uint  pattern = R(R_LINE_STIPPLE) & 0xffffu;
            const uint  factor  = clamp(R(R_LINE_STIPPLE) >> 16, 1u, 256u);
            const float along   = (i_shape2.w != 0.0) ? abs(win.x - i_shape.x) : abs(win.y - i_shape.y);
            const uint  counter = uint(max(i_shape2.x + along, 0.0));

            if (((pattern >> ((counter / factor) & 15u)) & 1u) == 0u)
                discard;
        }
    }
    /* Texturing, then the environment: unit 0's, then unit 1's on its
       result (ARB_multitexture's cascade) */
    if ((T(0u, 0u).x & 15u) != 0u)
        c = environment(0u, c, sample_texture(0u, st, st_dx, st_dy));
    if ((T(1u, 0u).x & 15u) != 0u)
        c = environment(1u, c, sample_texture(1u, st1, st1_dx, st1_dy));
    /* Fog: the fog distance's factor, f of the fragment's colour */
    if ((en & EN_FOG) != 0u) {
        const uint mode = R(R_FOG_MODE);
        float      f;

        if (mode == 0u) {
            const float range = RF(R_FOG_END) - RF(R_FOG_START);

            f = (range != 0.0) ? (RF(R_FOG_END) - fog_z) / range : 1.0;
        } else if (mode == 1u)
            f = exp(-RF(R_FOG_DENSITY) * fog_z);
        else {
            const float k = RF(R_FOG_DENSITY) * fog_z;

            f = exp(-k * k);
        }
        f     = clamp(f, 0.0, 1.0);
        c.rgb = mix(vec3(RF(R_FOG_COLOR), RF(R_FOG_COLOR + 1u), RF(R_FOG_COLOR + 2u)), c.rgb, f);
    }
    /* Smooth points and lines: their coverage multiplies alpha */
    if ((kind == KIND_POINTS) && (i_shape2.z != 0.0))
        c.a *= clamp(i_shape.z + 0.5 - distance(win, i_shape.xy), 0.0, 1.0);
    else if ((kind == KIND_LINES) && (i_shape2.z != 0.0)) {
        const vec2  dl  = i_shape.zw - i_shape.xy;
        const float len = length(dl);
        const vec2  u   = dl / len;
        const vec2  p   = win - i_shape.xy;
        const float t   = dot(p, u);
        const float s   = abs(dot(p, vec2(-u.y, u.x)));

        c.a *= clamp(i_shape2.y + 0.5 - s, 0.0, 1.0) * clamp(t + 0.5, 0.0, 1.0) * clamp(len - t + 0.5, 0.0, 1.0);
    }
    /* The alpha test, in 8-bit fixed point */
    if ((en & EN_ALPHA_TEST) != 0u) {
        const uint a   = uint(clamp(c.a, 0.0, 1.0) * 255.0 + 0.5);
        const uint ref = uint(clamp(RF(R_ALPHA_REF), 0.0, 1.0) * 255.0 + 0.5);

        if (!compare(R(R_ALPHA_FUNC), a, ref))
            discard;
    }
    o_color = c;
#ifdef COMPUTED_DEPTH
    {
        bool offset;

        if (kind == KIND_TRIANGLES)
            offset = (en & EN_OFFSET_FILL) != 0u;
        else if ((kind != KIND_PIXELS) && ((d.control & DRAW_POLYGON) != 0u))
            offset = (en & ((kind == KIND_LINES) ? EN_OFFSET_LINE : EN_OFFSET_POINT)) != 0u;
        else
            offset = false;
        put_depth(z, offset, slope);
    }
#endif
}
