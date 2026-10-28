/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the Glide device's pixel pipeline,
 *          one shader whose
 *          settings come from the uniform block, in a Voodoo's order:
 *          the TMUs' chain (TMU1 into TMU0), chroma key on the colour
 *          combine's "other" colour, the colour and alpha combine, the
 *          alpha test, fog, the depth value (Z, or W as the Voodoo's
 *          4.12 float, with the bias) and dithering to 5:6:5. The combine
 *          units compute in 8-bit integers as the hardware does:
 *          ((other - local) x (factor + 1)) >> 8, plus local or local
 *          alpha, clamped, inverted on request. Blending, the masks and
 *          the depth test are the pipeline's (vid_vgpu98_glide.c).
 *
 *          Each TMU samples as a Voodoo does: its LOD in the 256 level's
 *          texels, biased, clamped to the texture's levels; the level the
 *          LOD's integer part names (of the TMU's parity for a split
 *          texture); the magnification filter below the largest level,
 *          the minification filter otherwise; texel centres at +0.5;
 *          wrap by the coordinate's low bits, clamp or mirror. The host
 *          image holds the texture's levels decoded to 8:8:8:8, its level
 *          0 the largest; the texels are fetched and filtered here.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450

layout(std140, set = 0, binding = 0) uniform State {
    uvec4 comb;   /* colour combine, alpha combine, TMU0's and TMU1's texture combine (vid_vgpu98_glcmd.h) */
    uvec4 misc;   /* texture inverts (TMU0 bits 0-1, TMU1 bits 2-3), the constant ARGB, chroma (RGB, bit 24 on), alpha test (function | reference << 8) */
    uvec4 pix;    /* depth (mode | function << 8 | bias << 16), fog mode, fog colour RGB, F_* */
    uvec4 fog[4]; /* the fog table: 64 entries, four a word, the first in the low byte */
    uvec4 tex[2]; /* each TMU: T_* | s clamp << 8 | t clamp << 10; its levels (lodmin | lodmax << 4, the 256
                     level's width and height as log2 << 8 and << 12); the LOD bias (a float); detail
                     (bias, signed 8 | scale << 8 | max << 16) */
} u;

layout(set = 0, binding = 2) uniform sampler2D tex0;
layout(set = 0, binding = 3) uniform sampler2D tex1;

layout(location = 0) noperspective in vec4 v_color;
layout(location = 1) noperspective in vec4 v_depth;
layout(location = 2) noperspective in vec3 v_tex0;
layout(location = 3) noperspective in vec3 v_tex1;

layout(location = 0) out vec4 o_color;

#define F_DITHER_MASK 3u  /* 0 none, 1 2x2, 2 4x4 */
#define F_ITRGB       4u  /* grAlphaControlsITRGBLighting */
#define F_BLEND       8u  /* blending on: the colour goes out unquantized */
#define F_DEPTH       16u /* the depth value is written */

#define T_TEXTURE    1u   /* a texture is set */
#define T_EVEN       2u   /* the texture holds the even levels alone (Glide 2's numbers: 256, 64, 16, 4, 1) */
#define T_ODD        4u   /* the odd */
#define T_TRILINEAR  8u   /* grTexMipMapMode's lodBlend */
#define T_MIN_LINEAR 32u  /* bilinear minification */
#define T_MAG_LINEAR 64u  /* bilinear magnification */
#define T_LOD_DITHER 128u /* GR_MIPMAP_NEAREST_DITHER, allowed */

/* Glide's combine functions: zero other, subtract local, add (0 none, 1
   local, 2 local alpha). */
void
fn_bits(uint fn, out bool zero_other, out bool sub_local, out int add)
{
    zero_other = false;
    sub_local  = false;
    add        = 0;
    switch (fn) {
        case 0x0u: zero_other = true; break;
        case 0x1u: zero_other = true; add = 1; break;
        case 0x2u: zero_other = true; add = 2; break;
        case 0x3u: break;
        case 0x4u: add = 1; break;
        case 0x5u: add = 2; break;
        case 0x6u: sub_local = true; break;
        case 0x7u: sub_local = true; add = 1; break;
        case 0x8u: sub_local = true; add = 2; break;
        case 0x9u: zero_other = true; sub_local = true; add = 1; break;
        case 0x10u: zero_other = true; sub_local = true; add = 2; break;
        default: zero_other = true; break;
    }
}

/* One channel of a combine unit, 0-255 integers; m the factor select's
   value before the "one minus" bit. */
int
unit(uint fn, uint factor, int other, int local, int local_alpha, int m, bool invert)
{
    bool zero_other, sub_local;
    int  add;
    int  a, f, p;

    fn_bits(fn, zero_other, sub_local, add);
    a = (zero_other ? 0 : other) - (sub_local ? local : 0);
    f = (((factor & 8u) != 0u) ? 255 - m : m) + 1;
    p = (a * f) >> 8;
    p += (add == 1) ? local : (add == 2) ? local_alpha : 0;
    p = clamp(p, 0, 255);
    return invert ? 255 - p : p;
}

/* An iterated parameter's integer part as the Voodoo reads it: just below
   0 is 0, just above 255 is 255, others keep their low 8 bits. */
/* The Voodoo's iterators keep 12 fraction bits: an interpolated value a
   hair below an integer is that integer. */
#define ITER_EPS (1.0 / 4096.0)

int
iter8(float v)
{
    const int i = int(floor(v + ITER_EPS));

    return (i == -1) ? 0 : (i == 256) ? 255 : (i & 255);
}

ivec4
to8(vec4 c)
{
    return ivec4(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
}

/* 1/w as the Voodoo's 16-bit float: a 4-bit count of leading zeros of
   its fraction and the 12 bits after the leading one, inverted, plus one;
   0 at 1 and nearer, FFFFh past 2^-16. It grows with distance. */
uint
wfloat(float oow)
{
    uint f;
    uint e;
    uint mant;

    if (!(oow < 1.0))
        return 0u;
    if (!(oow >= 0.0))
        return 0xffffu;
    f = uint(oow * 65536.0);
    if (f == 0u)
        return 0xffffu;
    e    = 15u - uint(findMSB(f));
    mant = ((f << (e + 1u)) & 0xffffu) >> 4u;
    return min(((e << 12u) | (0xfffu - mant)) + 1u, 0xffffu);
}

uint
fog_entry(uint i)
{
    return (u.fog[i >> 4u][(i >> 2u) & 3u] >> ((i & 3u) * 8u)) & 0xffu;
}

/* The fog table at 1/w: the entry the float's top six bits pick, toward
   the next by the eight bits after them. */
int
fog_table(uint wf)
{
    const uint i    = (wf >> 10u) & 63u;
    const uint frac = (wf >> 2u) & 255u;
    const int  e0   = int(fog_entry(i));
    const int  e1   = (i < 63u) ? int(fog_entry(i + 1u)) : e0;

    return e0 + (((e1 - e0) * int(frac)) >> 8);
}

/* A coordinate inside a level of 2^size texels: clamp (1), mirror (2) or
   wrap (0) by its low bits. */
int
address(int c, int size, uint mode)
{
    const int n = 1 << size;

    if (mode == 1u)
        return clamp(c, 0, n - 1);
    if (mode == 2u)
        return ((c & n) != 0) ? (n - 1 - (c & (n - 1))) : (c & (n - 1));
    return c & (n - 1);
}

vec4
fetch(uint t, uint tx, int x, int y, int lw, int lh, int mip)
{
    const ivec2 at = ivec2(address(x, lw, (tx >> 8u) & 3u), address(y, lh, (tx >> 10u) & 3u));

    return (t == 0u) ? texelFetch(tex0, at, mip) : texelFetch(tex1, at, mip);
}

/* A TMU's texel, its clamped LOD (for the LOD fraction) and its LOD before
   the bias (for the detail factor). s and t are Glide's: the 256 level's
   texels, divided by 1/w per pixel; a negative 1/w reads coordinates 0. */
vec4
texel(uint t, vec3 stw, out float lod_clamped, out float lod_raw)
{
    const uvec4 tx     = u.tex[t];
    const vec2  st     = (stw.z > 0.0) ? stw.xy / stw.z : vec2(0.0);
    const vec2  dx     = dFdx(st);
    const vec2  dy     = dFdy(st);
    const float rho    = max(length(dx), length(dy));
    float       lambda = log2(max(rho, 1.0 / 65536.0));
    const int   lodmin = int(tx.y & 15u);
    const int   lodmax = int((tx.y >> 4u) & 15u);
    float       lod;
    bool        mag;
    int         lev, lw, lh, mip;
    vec2        p;

    /* Not a number (coordinates that are not) or past any level: kept to
       -16..16, so every integer taken of it is defined. */
    if (!(lambda >= -16.0))
        lambda = -16.0;
    if (lambda > 16.0)
        lambda = 16.0;
    lod = lambda + uintBitsToFloat(tx.z);
    if (!(lod >= -24.0) || (lod > 24.0))
        lod = lambda;

    lod_raw     = lambda;
    lod_clamped = clamp(lod, float(lodmin), float(lodmax));
    if ((tx.x & T_TEXTURE) == 0u)
        return vec4(1.0);
    mag = lod < float(lodmin);
    if ((tx.x & T_LOD_DITHER) != 0u) {
        const ivec2 q = ivec2(gl_FragCoord.xy) & 3;

        lod += float((q.y * 4 + q.x) * 3 & 15) * (0.75 / 15.0);
    }
    lod = clamp(lod, float(lodmin), float(lodmax));
    lev = int(floor(lod));
    if ((tx.x & (T_EVEN | T_ODD)) != 0u) {
        /* A split texture: the level of the TMU's parity at the LOD or
           next below it in size. */
        const int odd = ((tx.x & T_ODD) != 0u) ? 1 : 0;

        if ((lev & 1) != odd)
            lev++;
        if (lev > lodmax)
            lev -= 2;
        if (lev < lodmin)
            lev += 2;
    }
    mip = clamp(lev - lodmin, 0, lodmax - lodmin);
    lw  = max(int((tx.y >> 8u) & 15u) - lev, 0);
    lh  = max(int((tx.y >> 12u) & 15u) - lev, 0);
    p   = st * exp2(-float(lev));
    /* Coordinates past any texture's repeats (or not numbers) kept where an
       integer is defined of them; wrapping keeps their low bits. */
    if (!(p.x >= -16777216.0) || (p.x > 16777216.0))
        p.x = 0.0;
    if (!(p.y >= -16777216.0) || (p.y > 16777216.0))
        p.y = 0.0;
    if ((tx.x & (mag ? T_MAG_LINEAR : T_MIN_LINEAR)) == 0u) {
        const ivec2 i = ivec2(floor(p));

        return fetch(t, tx.x, i.x, i.y, lw, lh, mip);
    }
    {
        /* Bilinear: the 2x2 texels around the point, centres at +0.5. */
        const vec2  q = p - 0.5;
        const ivec2 i = ivec2(floor(q));
        const vec2  f = q - floor(q);
        const vec4  a = fetch(t, tx.x, i.x, i.y, lw, lh, mip);
        const vec4  b = fetch(t, tx.x, i.x + 1, i.y, lw, lh, mip);
        const vec4  c = fetch(t, tx.x, i.x, i.y + 1, lw, lh, mip);
        const vec4  d = fetch(t, tx.x, i.x + 1, i.y + 1, lw, lh, mip);

        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
}

/* A TMU's combine: its texel as local, the TMU upstream's output as other
   (none for TMU1). */
ivec4
tmu(uint t, ivec4 local, ivec4 other, float lod, float lod_raw)
{
    const uint  reg    = (t == 0u) ? u.comb.z : u.comb.w;
    const uint  inv    = (u.misc.x >> (t * 2u)) & 3u;
    const uvec4 tx     = u.tex[t];
    const uint  rgb_fn = reg & 0xffu;
    uint        rgb_f  = (reg >> 8u) & 0xffu;
    const uint  a_fn   = (reg >> 16u) & 0xffu;
    uint        a_f    = (reg >> 24u) & 0xffu;
    const int   frac   = int(fract(lod) * 256.0) & 255;
    int         detail;
    ivec4       r;
    int         k;

    /* Trilinear: the blend's direction turns over at every odd LOD, so one
       setting blends each level with the next. */
    if (((tx.x & T_TRILINEAR) != 0u) && ((int(floor(lod)) & 1) != 0)) {
        if ((rgb_f & 7u) == 5u)
            rgb_f ^= 8u;
        if ((a_f & 7u) == 5u)
            a_f ^= 8u;
    }
    /* The detail factor: (bias - LOD) << scale, at most detail max, the
       LOD before the bias in quarters. */
    {
        const int bias  = int((tx.w & 0xffu) ^ 0x80u) - 128;
        const int scale = int((tx.w >> 8u) & 7u);
        const int dmax  = int((tx.w >> 16u) & 0xffu);

        detail = min(dmax, clamp((bias - int(floor(lod_raw * 4.0))) << scale, 0, 255));
    }
    for (k = 0; k < 3; k++) {
        const uint sel = rgb_f & 7u;
        const int  m   = (sel == 1u) ? local[k] : (sel == 2u) ? other.a : (sel == 3u) ? local.a : (sel == 4u) ? detail :
                         (sel == 5u) ? frac : 0;

        r[k] = unit(rgb_fn, rgb_f, other[k], local[k], local.a, m, (inv & 1u) != 0u);
    }
    {
        const uint sel = a_f & 7u;
        const int  m   = (sel == 1u) ? local.a : (sel == 2u) ? other.a : (sel == 3u) ? local.a : (sel == 4u) ? detail :
                         (sel == 5u) ? frac : 0;

        r.a = unit(a_fn, a_f, other.a, local.a, local.a, m, (inv & 2u) != 0u);
    }
    return r;
}

bool
compare(uint fn, int a, int b)
{
    switch (fn) {
        case 0u: return false;
        case 1u: return a < b;
        case 2u: return a == b;
        case 3u: return a <= b;
        case 4u: return a > b;
        case 5u: return a != b;
        case 6u: return a >= b;
        default: return true;
    }
}

const int bayer4[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
const int bayer2[4]  = int[4](0, 2, 3, 1);

void
main()
{
    const ivec4 constant = ivec4((u.misc.y >> 16u) & 0xffu, (u.misc.y >> 8u) & 0xffu, u.misc.y & 0xffu, u.misc.y >> 24u);
    const ivec4 iter     = ivec4(iter8(v_color.r), iter8(v_color.g), iter8(v_color.b), iter8(v_color.a));
    const int   z_top    = (int(floor(v_depth.x + ITER_EPS)) >> 8) & 0xff; /* GR_COMBINE_LOCAL_DEPTH: Z's top 8 bits */
    const uint  cc       = u.comb.x;
    const uint  ac       = u.comb.y;
    float       lod0, lod1, raw0, raw1;
    vec4        s0, s1;
    ivec4       t1, t0, tex;
    ivec4       c_other, c_local;
    int         a_other, a_local;
    ivec4       out8;
    int         k;

    /* The TMUs: TMU1 first, its output TMU0's other (both sampled before
       anything else, where every pixel of a quad takes the same path). */
    s1  = texel(1u, v_tex1, lod1, raw1);
    s0  = texel(0u, v_tex0, lod0, raw0);
    t1  = tmu(1u, to8(s1), ivec4(0), lod1, raw1);
    t0  = tmu(0u, to8(s0), t1, lod0, raw0);
    tex = t0;

    /* The colour combine's inputs. */
    switch ((cc >> 20u) & 0xfu) {
        case 1u: c_other = tex; break;
        case 2u: c_other = constant; break;
        default: c_other = iter; break;
    }
    if ((u.pix.w & F_ITRGB) != 0u)
        c_local = ((tex.a & 0x80) != 0) ? constant : iter;
    else
        c_local = (((cc >> 16u) & 0xfu) == 1u) ? constant : iter;
    switch ((ac >> 20u) & 0xfu) {
        case 1u: a_other = tex.a; break;
        case 2u: a_other = constant.a; break;
        default: a_other = iter.a; break;
    }
    switch ((ac >> 16u) & 0xfu) {
        case 1u: a_local = constant.a; break;
        case 2u: a_local = z_top; break;
        default: a_local = iter.a; break;
    }

    /* Chroma key: the colour combine's other colour, 8:8:8, alpha aside. */
    if (((u.misc.z & 0x1000000u) != 0u) &&
        (uint((c_other.r << 16) | (c_other.g << 8) | c_other.b) == (u.misc.z & 0xffffffu)))
        discard;

    /* The colour and alpha combine units. */
    {
        const uint fn  = cc & 0xffu;
        const uint fac = (cc >> 8u) & 0xffu;
        const bool inv = ((cc >> 24u) & 1u) != 0u;

        for (k = 0; k < 3; k++) {
            const uint sel = fac & 7u;
            const int  m   = (sel == 1u) ? c_local[k] : (sel == 2u) ? a_other : (sel == 3u) ? a_local :
                             (sel == 4u) ? tex.a : (sel == 5u) ? tex[k] : 0;

            out8[k] = unit(fn, fac, c_other[k], c_local[k], a_local, m, inv);
        }
    }
    {
        const uint fn  = ac & 0xffu;
        const uint fac = (ac >> 8u) & 0xffu;
        const uint sel = fac & 7u;
        const int  m   = (sel == 1u) ? a_local : (sel == 2u) ? a_other : (sel == 3u) ? a_local : (sel == 4u) ? tex.a : 0;

        out8.a = unit(fn, fac, a_other, a_local, a_local, m, ((ac >> 24u) & 1u) != 0u);
    }

    /* The alpha test: the pixel's alpha against the reference. */
    if (!compare(u.misc.w & 7u, out8.a, int((u.misc.w >> 8u) & 0xffu)))
        discard;

    /* Fog: its alpha from the table at 1/w (or the fog coordinate),
       iterated alpha or Z, then the blend with the fog colour. */
    {
        const uint mode = u.pix.y & 0xffu;
        int        fa   = -1;

        if (mode == 1u)
            fa = clamp(int(floor(v_color.a)), 0, 255);
        else if (mode == 2u)
            fa = fog_table(wfloat(v_depth.y));
        else if (mode == 3u)
            fa = z_top;
        else if (mode == 4u)
            fa = fog_table(wfloat(v_depth.z));
        if (fa >= 0) {
            const ivec3 fc   = ivec3((u.pix.z >> 16u) & 0xffu, (u.pix.z >> 8u) & 0xffu, u.pix.z & 0xffu);
            const bool  mult = (u.pix.y & 0x100u) != 0u;
            const bool  add2 = (u.pix.y & 0x200u) != 0u;

            for (k = 0; k < 3; k++) {
                const int cin = out8[k];

                out8[k] = (mult && add2) ? 0 :
                          mult           ? ((fc[k] * (fa + 1)) >> 8) :
                          add2           ? ((cin * (255 - fa)) >> 8) :
                                           clamp((((fc[k] - cin) * (fa + 1)) >> 8) + cin, 0, 255);
            }
        }
    }

    /* The depth value: 16 bits, Z's integer part or W's float, plus the
       bias (the compare-to-bias modes compare the pixel's own). */
    if ((u.pix.w & F_DEPTH) != 0u) {
        const uint mode = u.pix.x & 0xffu;
        const int  bias = int((u.pix.x >> 16u) ^ 0x8000u) - 32768;
        int        z;

        if ((mode == 2u) || (mode == 4u))
            z = int(wfloat(v_depth.y));
        else
            z = clamp(int(floor(v_depth.x + ITER_EPS)), 0, 65535);
        if ((mode == 1u) || (mode == 2u))
            z = clamp(z + bias, 0, 65535);
        gl_FragDepth = float(z) / 65535.0;
    }

    /* To the 5:6:5 buffer: blended colour as it is (the blend rounds),
       else dithered or cut here, exactly. */
    if ((u.pix.w & F_BLEND) != 0u) {
        o_color = vec4(out8) / 255.0;
        return;
    }
    {
        const uint  dm = u.pix.w & F_DITHER_MASK;
        const ivec2 p  = ivec2(gl_FragCoord.xy);
        int         d  = 0; /* 0-15 */
        ivec3       q;

        if (dm == 2u)
            d = bayer4[(p.y & 3) * 4 + (p.x & 3)];
        else if (dm == 1u)
            d = bayer2[(p.y & 1) * 2 + (p.x & 1)] * 4;
        if (dm != 0u) {
            /* An ordered dither of 16 steps between two 5- (6-) bit values. */
            q.r = min((out8.r * 31 + d * 16) / 255, 31);
            q.g = min((out8.g * 63 + d * 16) / 255, 63);
            q.b = min((out8.b * 31 + d * 16) / 255, 31);
        } else
            q = ivec3(out8.r >> 3, out8.g >> 2, out8.b >> 3);
        o_color = vec4(float(q.r) / 31.0, float(q.g) / 63.0, float(q.b) / 31.0, float(out8.a) / 255.0);
    }
}
