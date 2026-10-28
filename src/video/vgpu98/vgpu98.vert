/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the vertex shader, command set 2.0's vertex
 *          pipeline (the project's spec/gpu-commands.md). It pulls each
 *          vertex from the draw's copy of its streams (a storage
 *          buffer) through the declaration the renderer resolved into
 *          the uniform block, then either takes a pre-transformed
 *          vertex as it is or runs the fixed-function transform, and
 *          maps it into Vulkan's clip space with Direct3D's pixel
 *          centres and the reference rasterizer's 1/16-pixel grid.
 *
 *          Most draws are plain: one invocation per vertex, Vulkan
 *          assembling the primitives. The shapes that need a whole
 *          primitive (lines, wireframe, point fill, WRAPn) read it from
 *          the draw's primitive list, one invocation per output vertex:
 *          a line becomes a band of quads that the fragment shader trims
 *          to exactly the pixels Direct3D's line rule draws, and a sized
 *          point or a point sprite (command set 2.1) a square.
 *
 *          Command set 2.1 adds DirectX 8's fixed function: the element
 *          types UBYTE4, SHORT2 and SHORT4, the point size, indexed vertex
 *          blending through the world matrix palette (a storage buffer),
 *          and tweening. Command set 2.2 adds vertex programs (PROGRAM 2):
 *          their instructions run here, over the input registers the
 *          declaration's INPUT elements load, into the record the fixed
 *          function fills.
 *
 *          The layout of the uniform block is mirrored by the renderer
 *          (vid_vgpu98_r3d.c, r3d_vubo_t).
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

/* What each invocation draws (the renderer's SHAPE_*). */
#define SHAPE_PLAIN  0u /* one vertex */
#define SHAPE_WRAP   1u /* a corner of a triangle with WRAPn */
#define SHAPE_POINTS 2u /* a vertex of a triangle in point fill */
#define SHAPE_WIRE   3u /* a corner of the band of a triangle's edge in wireframe */
#define SHAPE_LINE   4u /* a corner of the band of a line */
#define SHAPE_SPRITE 5u /* a corner of a point's square (2.1: sized points, sprites) */
#define SHAPE_FILLSPRITE 6u /* a corner of the square of a triangle's vertex in point fill (2.1) */

layout(constant_id = 0) const uint SHAPE   = 0u;
layout(constant_id = 1) const uint PROGRAM = 0u; /* 0: pre-transformed (POSITIONT), 1: fixed function (POSITION), 2: a vertex program (2.2) */

/* The declaration resolved: one slot per element use the pipeline reads. */
#define SLOT_POSITION  0u
#define SLOT_WEIGHT    1u
#define SLOT_NORMAL    2u
#define SLOT_DIFFUSE   3u
#define SLOT_SPECULAR  4u
#define SLOT_TEX0      5u
#define SLOT_PSIZE     13u /* 2.1 */
#define SLOT_INDICES   14u
#define SLOT_POSITION2 15u
#define SLOT_NORMAL2   16u
#define N_SLOTS        17u

/* Element types (VGPU98_DT_*) */
#define DT_FLOAT1   1u
#define DT_FLOAT4   4u
#define DT_D3DCOLOR 5u
#define DT_UBYTE4   6u /* 2.1 */
#define DT_SHORT2   7u
#define DT_SHORT4   8u

/* mode.x flags */
#define VF_LASTPIXEL 2u
#define VF_CLIPPING  4u

/* dx8.x flags (2.1) */
#define DX8_INDEXED 1u /* indexed vertex blending: the palette's matrices */
#define DX8_TWEEN   2u /* tweening */
#define DX8_SCALE   4u /* POINTSCALEENABLE */
#define DX8_SPRITE  8u /* POINTSPRITEENABLE */

/* The line band: three quads (start, middle, end) of two triangles. */
#define BAND_VERTICES 18u
/* A point's square: two triangles. */
#define SPRITE_VERTICES 6u

struct Light {
    vec4 diffuse;   /* rgb 0-1 */
    vec4 specular;
    vec4 ambient;
    vec4 position;  /* camera space; w: type (1 point, 2 spot, 3 directional) */
    vec4 direction; /* the direction to the light, camera space, normalized */
    vec4 atten;     /* attenuation 0, 1, 2, range squared */
    vec4 spot;      /* cos(theta/2), cos(phi/2), 1 / (cos(theta/2) - cos(phi/2)), falloff */
};

layout(std140, set = 0, binding = 3) uniform Vertex {
    uvec4 slot[N_SLOTS]; /* x: first word (of window vertex 0's element), y: stride in words, z: type (0 absent), w: end word */
    ivec4 range;         /* y: vertices in the draw's window: a window number from here up reads 0 */
    vec4  fb;            /* x: 2 / framebuffer width, y: 2 / framebuffer height, z: TEXTUREPERSPECTIVE */
    vec4  vp;            /* x: W / 2, y: H / 2, z: X + W / 2, w: Y + H / 2 */
    vec4  zrange;        /* x: minz, y: maxz - minz */
    uvec4 mode;          /* x: VF_*, y: WRAPn (set n: bits 4n to 4n + 3), z: CULLMODE, w: clip planes enabled */
    uvec4 tci[2];        /* stage s: TEXCOORDINDEX in tci[s / 4][s % 4] */
    uvec4 ttff[2];       /* stage s: TEXTURETRANSFORMFLAGS likewise */
    /* The fixed-function state (Direct3D's matrices as stored, so M * v
       is Direct3D's v * M). */
    mat4  wvp[4];        /* world j * view * projection */
    mat4  wv[4];         /* world j * view */
    mat4  wvinv[4];      /* (world j * view)^-1: normals are n * wvinv */
    mat4  tex[8];        /* the texture matrices */
    vec4  plane[6];      /* user clip planes, world space */
    mat4  world[4];      /* world j, for the clip planes */
    Light light[8];
    vec4  mat_diffuse;   /* the material's colours * 255 */
    vec4  mat_ambient;
    vec4  mat_specular;
    vec4  mat_emissive;
    vec4  material;      /* x: power, y: specular threshold, z: diffuse alpha (0-255), w: specular alpha */
    vec4  ambient;       /* the global ambient light, 0-1 */
    vec4  fog;           /* x: start, y: end, z: density, w: 255 / (end - start) or 0 */
    uvec4 tnl;           /* x: lights enabled (bits), y: TNL_* flags, z: blend weights used (0-3), w: vertex fog mode (0 none) */
    uvec4 sources;       /* the colour sources (0 material, 1 vertex diffuse, 2 vertex specular): x diffuse, y specular, z ambient, w emissive */
    /* Command set 2.1 */
    vec4  pt;            /* x: POINTSIZE, y: POINTSIZE_MIN, z: the largest size (POINTSIZE_MAX, at most 256), w: the viewport's height */
    vec4  ptscale;       /* x, y, z: POINTSCALE_A, B, C; w: TWEENFACTOR */
    uvec4 dx8;           /* x: DX8_* flags, y: the palette's entries (the last an identity world matrix), z, w: 0 */
    /* Command set 2.2 */
    uvec4 vprog;         /* x: the vertex program's instructions, y: the constants sent, z: the outputs it writes, w: VPF_* */
} vu;

/* vprog.w flags */
#define VPF_PIXEL 1u /* a pixel program is bound: coordinate sets padded as DirectX 8's texcoord */

/* 2.2: the vertex program's instructions and the vertex constants. */
layout(std430, set = 1, binding = 11) readonly buffer VProg {
    uvec4 ins[];
} vprog;

layout(std430, set = 1, binding = 13) readonly buffer VConst {
    vec4 c[];
} vconst;

layout(std430, set = 1, binding = 4) readonly buffer Verts {
    uint w[];
} vb;

/* The arena of resident buffers: a slot with SLOT_RES reads its stream
   there, whole, instead of in the draw's copy. */
#define SLOT_RES 0x100u

layout(std430, set = 0, binding = 10) readonly buffer Resident {
    uint w[];
} rb;

/* Word `at` of a slot's stream: the draw's copy's, or the arena's. */
uint
word_of(uint z, uint at)
{
    return ((z & SLOT_RES) != 0u) ? rb.w[at] : vb.w[at];
}

/* 2.1: the world matrix palette for indexed vertex blending, each entry
   world * view * projection, world * view, its inverse and world, as the
   uniform block's four (vid_vgpu98_r3d.c, r3d_pal_t). */
struct Pal {
    mat4 wvp;
    mat4 wv;
    mat4 wvinv;
    mat4 world;
};

layout(std430, set = 1, binding = 8) readonly buffer Palette {
    Pal p[];
} pal;

/* SHAPE_*: per primitive (v0, v1, v2, flags): a triangle's three vertices
   and its edge flags; a line's two and its LASTPIXEL. */
layout(std430, set = 1, binding = 5) readonly buffer Prims {
    ivec4 p[];
} pb;

layout(location = 0) out vec4 v_diffuse;
layout(location = 1) flat out vec4 v_diffuse_flat;
layout(location = 2) out vec4 v_specular;
layout(location = 3) flat out vec4 v_specular_flat;
layout(location = 4) out vec4 v_tc[8];               /* each stage's coordinates */
layout(location = 12) noperspective out vec2 v_zr;  /* screen z and rhw, linear in screen space */
layout(location = 13) flat out ivec4 v_line0;        /* the line rule: e0, e2, e1 low, e1 high */
layout(location = 14) flat out ivec4 v_line1;        /* first and last pixel on the major axis, x-major | step < 0 << 1 */
layout(location = 15) out float v_fog;               /* the fog factor, apart from the colours (2.2) */

out gl_PerVertex {
    vec4  gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[6];
};

/* A vertex after vertex processing. */
struct V {
    vec4  clip;     /* Direct3D clip space (fixed function), or (sx, sy, z, w) (pre-transformed) */
    vec2  scr;      /* screen position, before snapping: valid when on */
    bool  on;       /* w > 0: the vertex has a screen position */
    float w;        /* the w Vulkan interpolates with */
    vec2  zr;       /* screen z and rhw */
    vec4  diffuse;
    vec4  specular;
    vec4  tc[8];
    vec3  world;    /* world-space position (clip planes) */
    float dist;     /* camera-space distance from the eye (transformed vertices: point scaling) */
    float fog;      /* the fog factor, 0-1 */
    float psize;    /* a vertex program's point size (oPts) */
};

/* --- Fetching --------------------------------------------------------- */

/* An element type's words in the stream, and its components. */
uint
words(uint type)
{
    return (type <= DT_FLOAT4) ? type : (type == DT_SHORT4) ? 2u : 1u;
}

uint
comps(uint type)
{
    return (type <= DT_FLOAT4) ? type : (type == DT_SHORT2) ? 2u : (type == DT_D3DCOLOR) ? 1u : 4u;
}

/* A signed 16-bit integer from half a word, as a float. */
float
short_of(uint w, uint hi)
{
    return float(int(w << (16u - 16u * hi)) >> 16);
}

/* An element of window vertex v: its components (0 past the end of its
   stream's copy, or for a vertex outside the window), how many it has. */
vec4
fetch(uint s, int v, out uint n)
{
    uvec4 sl   = vu.slot[s];
    uint  type = sl.z & 0xffu;
    vec4  r    = vec4(0.0);

    n = (type != 0u) ? comps(type) : 0u;
    if ((type == 0u) || (v < 0) || (v >= vu.range.y))
        return r;
    uint64_t at64 = uint64_t(sl.x) + uint64_t(v) * uint64_t(sl.y);

    if (at64 + uint64_t(words(type)) > uint64_t(sl.w))
        return r;
    uint at = uint(at64);

    if (type == DT_D3DCOLOR) {
        uint c = word_of(sl.z, at);

        return vec4(float((c >> 16u) & 255u), float((c >> 8u) & 255u), float(c & 255u), float(c >> 24u)) / 255.0;
    }
    if (type == DT_UBYTE4) {
        uint c = word_of(sl.z, at);

        return vec4(float(c & 255u), float((c >> 8u) & 255u), float((c >> 16u) & 255u), float(c >> 24u));
    }
    if (type == DT_SHORT2)
        return vec4(short_of(word_of(sl.z, at), 0u), short_of(word_of(sl.z, at), 1u), 0.0, 0.0);
    if (type == DT_SHORT4)
        return vec4(short_of(word_of(sl.z, at), 0u), short_of(word_of(sl.z, at), 1u), short_of(word_of(sl.z, at + 1u), 0u),
                    short_of(word_of(sl.z, at + 1u), 1u));
    for (uint i = 0u; i < n; i++)
        r[i] = uintBitsToFloat(word_of(sl.z, at + i));
    return r;
}

/* A colour: the element's, or `dflt` when the declaration has none. */
vec4
fetch_color(uint s, int v, vec4 dflt)
{
    uint n;
    vec4 c = fetch(s, v, n);

    return (vu.slot[s].z != 0u) ? c : dflt;
}

/* A texture coordinate set padded to four: 1 at the first missing
   component, 0 after it; a set the declaration lacks is (0, 0, 0, 0). */
vec4
fetch_set(uint set, int v)
{
    uint n;
    vec4 c = fetch(SLOT_TEX0 + set, v, n);

    if (n == 0u)
        return vec4(0.0);
    if (n < 4u)
        c[n] = 1.0;
    return c;
}

/* A coordinate set as a pixel program reads it (2.2): missing components 0
   and the fourth 1, as DirectX 8's texcoord pads them (not the transform's
   padding); a set the declaration lacks is (0, 0, 0, 0). */
vec4
fetch_set_dx(uint set, int v)
{
    uint n;
    vec4 c = fetch(SLOT_TEX0 + set, v, n);

    if (n == 0u)
        return vec4(0.0);
    if (n < 4u)
        c.w = 1.0;
    return c;
}

/* --- Screen space --------------------------------------------------------- */

/* The reference rasterizer's grid: n.4, rounded to even, within 2^22 pixels
   (NaN to the low end). */
vec2
clamp_grid(vec2 s)
{
    if (!(s.x > -4194304.0))
        s.x = -4194304.0;
    else if (s.x > 4194304.0)
        s.x = 4194304.0;
    if (!(s.y > -4194304.0))
        s.y = -4194304.0;
    else if (s.y > 4194304.0)
        s.y = 4194304.0;
    return s;
}

vec2
snap(vec2 s)
{
    return roundEven(clamp_grid(s) * 16.0) / 16.0;
}

ivec2
snap16(vec2 s)
{
    return ivec2(roundEven(clamp_grid(s) * 16.0));
}

/* Vulkan clip space for a screen position (pixel centres at whole
   numbers) at w. */
vec4
to_clip(vec2 s, float z_ndc, float w)
{
    vec2 ndc = (s + 0.5) * vu.fb.xy - 1.0;

    return vec4(ndc * w, z_ndc * w, w);
}

/* A transformed vertex's Vulkan position: the viewport mapping of its clip
   space, snapped when it has a screen position on the grid, linear in clip
   space when it does not (behind the eye, or so far out that the grid's
   clamp would move it: the host GPU then clips it where it really is). */
vec4
place(V a)
{
    if (PROGRAM == 0u)
        return to_clip(snap(a.scr), clamp(a.clip.z, 0.0, 1.0), a.w);
    if (a.on)
        return to_clip(snap(a.scr), a.clip.z / a.clip.w, a.clip.w);
    /* (sx + 0.5) * 2 / fw - 1 at any w, sx = X + W/2 + x/w * W/2; y down. */
    return vec4(a.clip.x * vu.vp.x * vu.fb.x + a.clip.w * ((vu.vp.z + 0.5) * vu.fb.x - 1.0),
                -a.clip.y * vu.vp.y * vu.fb.y + a.clip.w * ((vu.vp.w + 0.5) * vu.fb.y - 1.0), a.clip.z, a.clip.w);
}

/* The screen position and z of a transformed vertex from its clip space. */
void
project(inout V a)
{
    if (PROGRAM == 0u)
        return;
    a.on = a.clip.w > 0.0;
    a.w  = a.clip.w;
    if (a.on) {
        float rw = 1.0 / a.clip.w;

        a.scr = vec2(vu.vp.z + a.clip.x * rw * vu.vp.x, vu.vp.w - a.clip.y * rw * vu.vp.y);
        a.zr  = vec2(vu.zrange.x + a.clip.z * rw * vu.zrange.y, rw);
        a.on  = all(lessThanEqual(abs(a.scr), vec2(4194304.0)));
    } else
        a.zr = vec2(vu.zrange.x, 0.0);
}

/* --- Vertex processing ---------------------------------------------------- */

uint
stage_tci(uint s)
{
    return vu.tci[s >> 2u][s & 3u];
}

uint
stage_ttff(uint s)
{
    return vu.ttff[s >> 2u][s & 3u];
}

/* A pre-transformed vertex: screen space as it comes (spec/gpu-commands.md). */
V
process_pretransformed(int v)
{
    V    a;
    uint n;
    vec4 p   = fetch(SLOT_POSITION, v, n);
    float rhw = p.w;

    a.clip     = p;
    a.scr      = p.xy;
    a.on       = true;
    /* Perspective-correct interpolation divides by w; with TEXTUREPERSPECTIVE
       off every attribute is interpolated linearly in screen space. */
    a.w        = ((vu.fb.z != 0.0) && (rhw > 0.0)) ? (1.0 / rhw) : 1.0;
    a.zr       = vec2(p.z, rhw);
    a.diffuse  = fetch_color(SLOT_DIFFUSE, v, vec4(1.0));
    a.specular = fetch_color(SLOT_SPECULAR, v, vec4(0.0));
    a.world    = vec3(0.0);
    a.dist     = 0.0;
    a.fog      = a.specular.a; /* a pre-transformed vertex's fog factor is its specular alpha */
    a.psize    = 0.0;
    /* No generation and no transform for pre-transformed vertices; the
       fragment stage still divides PROJECTED coordinates. */
    for (uint s = 0u; s < 8u; s++)
        a.tc[s] = ((vu.vprog.w & VPF_PIXEL) != 0u) ? fetch_set_dx(stage_tci(s) & 7u, v) : fetch_set(stage_tci(s) & 7u, v);
    return a;
}

/* tnl.y flags */
#define TNL_LIGHTING  1u
#define TNL_SPECULAR  2u
#define TNL_LOCAL     4u
#define TNL_NORMALIZE 8u
#define TNL_RANGEFOG  16u

/* Light types (Light.position.w) */
#define LIGHT_SPOT        2.0
#define LIGHT_DIRECTIONAL 3.0

#define FLT_MAX 3.402823466e+38

/* C's pow where lighting meets it: 1 for a zero power, 0 for a base at or
   below 0. C gives 0 for a zero base with a positive power, NaN for a
   negative base (the reference T&L's colour then comes out 0), and infinity
   for a zero base with a negative power (only a negative material power
   meets that; the term is dropped here). */
float
pow0(float x, float y)
{
    if (y == 0.0)
        return 1.0;
    return (x > 0.0) ? pow(x, y) : 0.0;
}

/* The reference T&L's FTOI, a conversion in the FPU's rounding mode, which
   is round to nearest (halves to even) when it runs, clamped to a colour
   byte; NaN and negative values give 0 (spec/gpu-commands.md). */
float
byte_of(float x)
{
    if (!(x > 0.0))
        return 0.0;
    return (x >= 255.0) ? 255.0 : roundEven(x);
}

/* A colour element as its bytes (0-255), or `dflt` when the declaration has
   none. */
vec4
fetch_bytes(uint s, int v, vec4 dflt)
{
    uint n;
    vec4 c = fetch(s, v, n);

    return (vu.slot[s].z != 0u) ? round(c * 255.0) : dflt;
}

vec3
color_source(uint src, vec3 mat, vec4 vd, vec4 vs)
{
    return (src == 1u) ? vd.rgb : (src == 2u) ? vs.rgb : mat;
}

float
alpha_source(uint src, float mat, vec4 vd, vec4 vs)
{
    return (src == 1u) ? vd.a : (src == 2u) ? vs.a : mat;
}

/* Direct3D's lighting, as the reference T&L does it (spec/gpu-commands.md,
   "Vertex processing"): colours in units of 0-255, the results truncated to
   bytes. `vd` and `vs` are the vertex's colours as bytes. */
void
light_vertex(vec3 eye, vec3 nrm, bool has_n, vec4 vd, vec4 vs, out vec4 dout, out vec4 sout)
{
    vec3 amb_src = color_source(vu.sources.z, vu.mat_ambient.rgb, vd, vs);
    vec3 dif_src = color_source(vu.sources.x, vu.mat_diffuse.rgb, vd, vs);
    vec3 spe_src = color_source(vu.sources.y, vu.mat_specular.rgb, vd, vs);
    vec3 emi_src = color_source(vu.sources.w, vu.mat_emissive.rgb, vd, vs);
    bool do_spec = (vu.tnl.y & TNL_SPECULAR) != 0u;
    vec3 E       = ((vu.tnl.y & TNL_LOCAL) != 0u) ? normalize(-eye) : vec3(0.0, 0.0, -1.0);
    vec3 dif     = vu.ambient.rgb * amb_src + emi_src;
    vec3 spe     = vec3(0.0);

    for (uint i = 0u; i < 8u; i++) {
        if (((vu.tnl.x >> i) & 1u) == 0u)
            continue;
        if (vu.light[i].position.w == LIGHT_DIRECTIONAL) {
            vec3  L = vu.light[i].direction.xyz;
            float d;

            dif += vu.light[i].ambient.rgb * amb_src;
            if (!has_n)
                continue;
            d = dot(L, nrm);
            if (d > 0.0) {
                dif += vu.light[i].diffuse.rgb * dif_src * d;
                if (do_spec) {
                    vec3  h  = normalize(L + E);
                    float d2 = dot(h, nrm);

                    if ((d2 > 0.0) && (d2 >= vu.material.y))
                        spe += vu.light[i].specular.rgb * spe_src * pow0(d2, vu.material.x);
                }
            }
        } else {
            vec3  D     = vu.light[i].position.xyz - eye;
            float dist2 = dot(D, D);
            float dist;
            float att;
            float rd;
            float d;

            /* At or beyond the range, or exactly at the light: nothing. */
            if ((dist2 >= vu.light[i].atten.w) || (dist2 == 0.0))
                continue;
            dist = sqrt(dist2);
            att  = vu.light[i].atten.x + vu.light[i].atten.y * dist + vu.light[i].atten.z * dist2;
            att  = (att == 0.0) ? FLT_MAX : (1.0 / att);
            rd   = 1.0 / dist;
            if (vu.light[i].position.w == LIGHT_SPOT) {
                float cone = dot(D, vu.light[i].direction.xyz) * rd;

                if (cone <= vu.light[i].spot.y)
                    continue;
                if (cone < vu.light[i].spot.x) {
                    float val = (cone - vu.light[i].spot.y) * vu.light[i].spot.z;

                    if (vu.light[i].spot.w - 1.0 != 0.0)
                        val = pow0(val, vu.light[i].spot.w);
                    att *= val;
                }
            }
            dif += att * vu.light[i].ambient.rgb * amb_src;
            if (!has_n)
                continue;
            d = dot(D, nrm);
            if (d > 0.0) {
                d *= rd * att;
                dif += vu.light[i].diffuse.rgb * dif_src * d;
                if (do_spec) {
                    vec3  h  = normalize(D * rd + E);
                    float d2 = dot(h, nrm);

                    if (d2 >= vu.material.y)
                        spe += vu.light[i].specular.rgb * spe_src * (pow0(d2, vu.material.x) * att);
                }
            }
        }
    }

    dout = vec4(byte_of(dif.r), byte_of(dif.g), byte_of(dif.b), alpha_source(vu.sources.x, vu.material.z, vd, vs));
    if (do_spec)
        sout = vec4(byte_of(spe.r), byte_of(spe.g), byte_of(spe.b), alpha_source(vu.sources.y, vu.material.w, vd, vs));
    else if (vu.slot[SLOT_SPECULAR].z != 0u)
        sout = vs; /* the vertex's own, alpha included */
    else
        sout = vec4(0.0, 0.0, 0.0, alpha_source(vu.sources.y, vu.material.w, vd, vs));
}

/* Vertex fog: the factor in the specular alpha, as a byte (the reference
   T&L stores the low 8 bits of the value converted to an integer, rounded as
   byte_of rounds). */
float
fog_vertex(vec3 eye)
{
    float d = ((vu.tnl.y & TNL_RANGEFOG) != 0u) ? length(eye) : eye.z;
    float f;

    if (vu.tnl.w == 3u) {
        if (d < vu.fog.x)
            return 255.0;
        if (d >= vu.fog.y)
            return 0.0;
        f = (vu.fog.y - d) * vu.fog.w;
    } else {
        float t = d * vu.fog.z;

        if (vu.tnl.w == 2u)
            t *= t;
        f = exp(-t) * 255.0;
    }
    f = roundEven(f);
    if (!(f >= -2147483648.0) || !(f < 2147483648.0))
        return 0.0; /* the x87's integer indefinite: its low byte is 0 */
    return float(uint(int(f)) & 255u);
}

/* The reflection vector for generated coordinates. */
vec3
reflection(vec3 eye, vec3 nrm)
{
    if ((vu.tnl.y & TNL_LOCAL) != 0u) {
        vec3 e = eye * (1.0 / sqrt(dot(eye, eye)));

        return e - nrm * (2.0 * dot(e, nrm));
    }
    return vec3(-nrm.x * (2.0 * nrm.z), -nrm.y * (2.0 * nrm.z), 1.0 - nrm.z * (2.0 * nrm.z));
}

/* A vertex with an untransformed position, through the fixed-function
   pipeline: tweening or blending (indexed through the palette, 2.1),
   lighting, fog, generated coordinates and texture transforms. */
V
process_fixed(int v)
{
    V     a;
    uint  n;
    vec4  p     = vec4(fetch(SLOT_POSITION, v, n).xyz, 1.0);
    vec4  bw    = fetch(SLOT_WEIGHT, v, n);
    vec3  nin   = fetch(SLOT_NORMAL, v, n).xyz;
    bool  has_n = vu.slot[SLOT_NORMAL].z != 0u;
    uint  nb    = vu.tnl.z;
    float cumul = 0.0;
    vec3  eye   = vec3(0.0);
    vec3  nrm   = vec3(0.0);
    vec4  vd    = fetch_bytes(SLOT_DIFFUSE, v, vec4(255.0));
    vec4  vs    = fetch_bytes(SLOT_SPECULAR, v, vec4(0.0));
    vec4  dif;
    vec4  spe;

    /* Tweening (2.1): between the two positions, and the two normals,
       where the declaration has the second, before anything else; world
       matrix 0 then (nb is 0). */
    if ((vu.dx8.x & DX8_TWEEN) != 0u) {
        float t = vu.ptscale.w;

        if (vu.slot[SLOT_POSITION2].z != 0u)
            p.xyz = p.xyz * (1.0 - t) + fetch(SLOT_POSITION2, v, n).xyz * t;
        if (vu.slot[SLOT_NORMAL2].z != 0u)
            nin = nin * (1.0 - t) + fetch(SLOT_NORMAL2, v, n).xyz * t;
    }

    a.clip  = vec4(0.0);
    a.world = vec3(0.0);
    if ((vu.dx8.x & DX8_INDEXED) != 0u) {
        /* Indexed vertex blending (2.1): matrix j is the palette's entry
           of the vertex's index j (an index the palette was not sent for
           is its last entry, an identity world matrix). A declaration
           without BLENDINDICES takes indices 0, 1, 2 and 3, as Direct3D 8
           has blending without indices. */
        vec4 ix = ((vu.slot[SLOT_INDICES].z & 0xffu) != 0u) ? fetch(SLOT_INDICES, v, n) : vec4(0.0, 1.0, 2.0, 3.0);

        for (uint j = 0u; j <= nb; j++) {
            uint  e = min(uint(ix[j]), vu.dx8.y - 1u);
            float b;

            if (nb == 0u)
                b = 1.0;
            else if (j == nb)
                b = 1.0 - cumul;
            else {
                b = bw[j];
                cumul += b;
            }
            a.clip += (pal.p[e].wvp * p) * b;
            eye += (pal.p[e].wv * p).xyz * b;
            nrm += (vec4(nin, 0.0) * pal.p[e].wvinv).xyz * b;
            a.world += (pal.p[e].world * p).xyz * b;
        }
    } else {
        for (uint j = 0u; j <= nb; j++) {
            float b;

            if (nb == 0u)
                b = 1.0;
            else if (j == nb)
                b = 1.0 - cumul;
            else {
                b = bw[j];
                cumul += b;
            }
            a.clip += (vu.wvp[j] * p) * b;
            eye += (vu.wv[j] * p).xyz * b;
            nrm += (vec4(nin, 0.0) * vu.wvinv[j]).xyz * b;
            a.world += (vu.world[j] * p).xyz * b;
        }
    }
    a.dist = length(eye);
    if ((vu.tnl.y & TNL_NORMALIZE) != 0u) {
        float l2 = dot(nrm, nrm);

        if (l2 > 0.0)
            nrm *= 1.0 / sqrt(l2);
    }

    if ((vu.tnl.y & TNL_LIGHTING) != 0u)
        light_vertex(eye, nrm, has_n, vd, vs, dif, spe);
    else {
        dif = vd;
        spe = vs;
    }
    /* The fog factor travels apart (2.2): vertex fog's, or without it the
       specular alpha, as before; the specular alpha stays the colour's. */
    a.fog      = ((vu.tnl.w != 0u) ? fog_vertex(eye) : spe.a) / 255.0;
    a.psize    = 0.0;
    a.diffuse  = dif / 255.0;
    a.specular = spe / 255.0;

    for (uint s = 0u; s < 8u; s++) {
        uint tci = stage_tci(s);
        uint gen = (tci >> 16u) & 3u;
        vec4 c;

        if (gen == 1u)
            c = vec4(nrm, 1.0);
        else if (gen == 2u)
            c = vec4(eye, 1.0);
        else if (gen == 3u)
            c = vec4(reflection(eye, nrm), 1.0);
        else if (((vu.vprog.w & VPF_PIXEL) != 0u) && ((stage_ttff(s) & 0xffu) == 0u))
            c = fetch_set_dx(tci & 7u, v); /* a pixel program's untransformed set (2.2) */
        else
            c = fetch_set(tci & 7u, v);
        if ((stage_ttff(s) & 0xffu) != 0u) {
            uint n = stage_ttff(s) & 0xffu;

            c = vu.tex[s] * c;
            /* A pixel program reads the COUNT coordinates the transform
               passes, the others as DirectX 8's texcoord pads them (0, the
               fourth 1). */
            if (((vu.vprog.w & VPF_PIXEL) != 0u) && (n < 4u))
                c = mix(c, vec4(0.0, 0.0, 0.0, 1.0), greaterThanEqual(uvec4(0u, 1u, 2u, 3u), uvec4(n)));
        }
        a.tc[s] = c;
    }
    project(a);
    return a;
}

/* --- Vertex programs (command set 2.2) ------------------------------------ */

/* Instruction fields and files (vid_vgpu98_prog.h, spec/gpu-commands.md). */
#define PF_TEMP   1u
#define PF_INPUT  2u
#define PF_CONST  3u
#define PF_ADDR   4u
#define PF_OUTPUT 6u
#define PF_FIXED  8u
#define INS_SAT   0x01000000u
#define SRC_REL   0x01000000u

#define PI_MOV  0x01u
#define PI_ADD  0x02u
#define PI_SUB  0x03u
#define PI_MUL  0x04u
#define PI_MAD  0x05u
#define PI_DP3  0x06u
#define PI_DP4  0x07u
#define PI_LRP  0x08u
#define PI_CND  0x09u
#define PI_CMP  0x0au
#define PI_MIN  0x0bu
#define PI_MAX  0x0cu
#define PI_SLT  0x0du
#define PI_SGE  0x0eu
#define PI_RCP  0x0fu
#define PI_RSQ  0x10u
#define PI_EXP  0x11u
#define PI_LOG  0x12u
#define PI_EXPP 0x13u
#define PI_LOGP 0x14u
#define PI_FRC  0x15u
#define PI_LIT  0x16u
#define PI_DST  0x17u
#define PI_MOVA 0x18u

/* Outputs */
#define PO_POS 0u
#define PO_D0  1u
#define PO_D1  2u
#define PO_FOG 3u
#define PO_PTS 4u
#define PO_T0  5u

vec4 pv_in[16];  /* the input registers */
vec4 pv_r[16];   /* temporaries */
vec4 pv_o[13];   /* outputs */
int  pv_a0 = 0;  /* the address register */

/* A fixed vector: 0, 1, (0, 0, 0, 1), 0.5. */
vec4
fixed_vec(uint n)
{
    return (n == 1u) ? vec4(1.0) : (n == 2u) ? vec4(0.0, 0.0, 0.0, 1.0) : (n == 3u) ? vec4(0.5) : vec4(0.0);
}

/* A source modifier (DirectX 8's codes), after the swizzle. */
vec4
apply_mod(vec4 x, uint m)
{
    switch (m) {
        case 1u: return -x;
        case 2u: return x - 0.5;
        case 3u: return 0.5 - x;
        case 4u: return 2.0 * x - 1.0;
        case 5u: return 1.0 - 2.0 * x;
        case 6u: return 1.0 - x;
        case 7u: return 2.0 * x;
        case 8u: return -2.0 * x;
        case 9u: return vec4((x.z == 0.0) ? 1.0 : x.x / x.z, (x.z == 0.0) ? 1.0 : x.y / x.z, x.z, x.w);
        case 10u: return vec4((x.w == 0.0) ? 1.0 : x.x / x.w, (x.w == 0.0) ? 1.0 : x.y / x.w, x.z, x.w);
        default: return x;
    }
}

vec4
swizzle(vec4 x, uint swz)
{
    return vec4(x[swz & 3u], x[(swz >> 2u) & 3u], x[(swz >> 4u) & 3u], x[(swz >> 6u) & 3u]);
}

/* A source operand of a vertex program. */
vec4
pv_src(uint s)
{
    uint file = s & 15u;
    uint reg  = (s >> 4u) & 255u;
    vec4 x;

    if (file == PF_TEMP)
        x = pv_r[reg & 15u];
    else if (file == PF_INPUT)
        x = pv_in[reg & 15u];
    else if (file == PF_CONST) {
        /* A relative register outside the constants sent reads 0. */
        int k = int(reg) + (((s & SRC_REL) != 0u) ? pv_a0 : 0);

        x = ((k >= 0) && (k < int(vu.vprog.y))) ? vconst.c[k] : vec4(0.0);
    } else
        x = fixed_vec(reg);
    return apply_mod(swizzle(x, (s >> 12u) & 255u), (s >> 20u) & 15u);
}

#define FLT_MAX_P 3.402823466e+38
#define INF_P     uintBitsToFloat(0x7f800000u)

/* An arithmetic operation (spec/gpu-commands.md, Command set 2.2). */
vec4
op_eval(uint op, vec4 a, vec4 b, vec4 c)
{
    switch (op) {
        case PI_MOV: return a;
        case PI_ADD: return a + b;
        case PI_SUB: return a - b;
        case PI_MUL: return a * b;
        case PI_MAD: return a * b + c;
        case PI_DP3: return vec4(dot(a.xyz, b.xyz));
        case PI_DP4: return vec4(dot(a, b));
        case PI_LRP: return a * b + (1.0 - a) * c;
        case PI_CND: return mix(c, b, greaterThan(a, vec4(0.5)));
        case PI_CMP: return mix(c, b, greaterThanEqual(a, vec4(0.0)));
        case PI_MIN: return mix(b, a, lessThan(a, b));
        case PI_MAX: return mix(b, a, greaterThanEqual(a, b));
        case PI_SLT: return vec4(lessThan(a, b));
        case PI_SGE: return vec4(greaterThanEqual(a, b));
        case PI_RCP: return vec4((a.x == 1.0) ? 1.0 : (a.x == 0.0) ? INF_P : 1.0 / a.x);
        case PI_RSQ: {
            float v = abs(a.x);

            return vec4((v == 1.0) ? 1.0 : (v == 0.0) ? INF_P : inversesqrt(v));
        }
        case PI_EXP: return vec4(exp2(a.x));
        case PI_LOG: {
            float v = abs(a.x);

            return vec4((v == 0.0) ? -FLT_MAX_P : log2(v));
        }
        case PI_EXPP: {
            float f = floor(a.x);

            return vec4(exp2(f), a.x - f, uintBitsToFloat(floatBitsToUint(exp2(a.x)) & 0xffffff00u), 1.0);
        }
        case PI_LOGP: {
            float v = abs(a.x);

            if (v == 0.0)
                return vec4(-FLT_MAX_P, 1.0, -FLT_MAX_P, 1.0);
            return vec4(float(int((floatBitsToUint(v) >> 23u) & 255u) - 127),
                        uintBitsToFloat((floatBitsToUint(v) & 0x7fffffu) | 0x3f800000u),
                        uintBitsToFloat(floatBitsToUint(log2(v)) & 0xffffff00u), 1.0);
        }
        case PI_FRC: return a - floor(a);
        case PI_LIT: {
            float p = clamp(a.w, -127.9961, 127.9961);

            return vec4(1.0, max(a.x, 0.0), ((a.x > 0.0) && (a.y > 0.0)) ? pow(a.y, p) : 0.0, 1.0);
        }
        case PI_DST: return vec4(1.0, a.y * b.y, a.z, b.w);
        default: return vec4(0.0);
    }
}

/* Input register n of vertex v: its INPUT element as its type reads (the
   floats and SHORT2 with w 1, the others whole), (0, 0, 0, 1) without one. */
vec4
program_input(uint n, int v)
{
    uint t = vu.slot[n].z & 0xffu;
    uint k;
    vec4 c;

    if (t == 0u)
        return vec4(0.0, 0.0, 0.0, 1.0);
    c = fetch(n, v, k);
    if ((t <= DT_FLOAT1 + 2u) || (t == DT_SHORT2))
        c.w = 1.0;
    return c;
}

/* A vertex through its vertex program: the outputs into the record the
   fixed function fills. */
V
process_program(int v)
{
    V    a;
    uint i;

    for (i = 0u; i < 16u; i++) {
        pv_in[i] = program_input(i, v);
        pv_r[i]  = vec4(0.0);
    }
    pv_o[PO_POS] = vec4(0.0);
    pv_o[PO_D0]  = vec4(1.0);
    pv_o[PO_D1]  = vec4(0.0);
    pv_o[PO_FOG] = vec4(1.0);
    pv_o[PO_PTS] = vec4(vu.pt.x);
    for (i = 0u; i < 8u; i++)
        pv_o[PO_T0 + i] = vec4(0.0, 0.0, 0.0, 1.0);
    pv_a0 = 0;

    for (i = 0u; i < vu.vprog.x; i++) {
        uvec4 in_ = vprog.ins[i];
        uint  op  = in_.x & 255u;
        vec4  r   = op_eval(op, pv_src(in_.y), pv_src(in_.z), pv_src(in_.w));
        uint  df  = (in_.x >> 8u) & 15u;
        uint  dr  = (in_.x >> 12u) & 255u;
        uint  m   = (in_.x >> 20u) & 15u;
        bvec4 wm  = bvec4((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u, (m & 8u) != 0u);

        if (op == PI_MOVA) {
            pv_a0 = int(floor(pv_src(in_.y).x));
            continue;
        }
        if ((in_.x & INS_SAT) != 0u)
            r = clamp(r, 0.0, 1.0);
        if (df == PF_OUTPUT)
            pv_o[min(dr, 12u)] = mix(pv_o[min(dr, 12u)], r, wm);
        else
            pv_r[dr & 15u] = mix(pv_r[dr & 15u], r, wm);
    }

    a.clip     = pv_o[PO_POS];
    a.scr      = vec2(0.0);
    a.on       = false;
    a.w        = 1.0;
    a.zr       = vec2(0.0);
    a.diffuse  = clamp(pv_o[PO_D0], 0.0, 1.0);
    a.specular = clamp(pv_o[PO_D1], 0.0, 1.0);
    a.fog      = clamp(pv_o[PO_FOG].x, 0.0, 1.0);
    a.psize    = pv_o[PO_PTS].x;
    for (i = 0u; i < 8u; i++)
        a.tc[i] = pv_o[PO_T0 + i];
    a.world = vec3(0.0);
    a.dist  = 0.0;
    project(a);
    return a;
}

V
process(int v)
{
    return (PROGRAM == 0u) ? process_pretransformed(v) : (PROGRAM == 2u) ? process_program(v) : process_fixed(v);
}

/* A vertex's distance from user clip plane i: in world space, or in clip
   space for a vertex program's (2.2), as DirectX 8 has them then. */
float
plane_dist(V a, uint i)
{
    return (PROGRAM == 2u) ? dot(a.clip, vu.plane[i]) : dot(vec4(a.world, 1.0), vu.plane[i]);
}

/* The user clip planes CLIPPLANEENABLE selects, with CLIPPING, for
   transformed vertices: Vulkan clips where a distance is negative. */
void
clip_distances(V a)
{
    for (uint i = 0u; i < 6u; i++) {
        bool on = (PROGRAM != 0u) && ((vu.mode.x & VF_CLIPPING) != 0u) && (((vu.mode.w >> i) & 1u) != 0u);

        gl_ClipDistance[i] = on ? plane_dist(a, i) : 1.0;
    }
}

/* Nothing of this invocation's primitive is drawn. */
void
kill()
{
    gl_Position  = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0; /* point fill draws point lists */
    for (uint i = 0u; i < 6u; i++)
        gl_ClipDistance[i] = -1.0;
}

/* --- WRAPn ---------------------------------------------------------------- */

/* The shorter way around from a to b. */
float
wrap_diff(float b, float a)
{
    float d1 = b - a;
    float d2 = (d1 < 0.0) ? d1 + 1.0 : d1 - 1.0;

    return (abs(d1) < abs(d2)) ? d1 : d2;
}

/* WRAPn of each stage's coordinate set (bits 0-3: its first to fourth
   coordinate), for vertex b of a primitive whose first vertex is a. */
void
wrap(V a, inout V b)
{
    for (uint s = 0u; s < 8u; s++) {
        uint set  = (PROGRAM == 2u) ? s : (stage_tci(s) & 7u); /* a vertex program's stage s reads oTs (2.2) */
        uint bits = (vu.mode.y >> (4u * set)) & 15u;

        for (uint c = 0u; c < 4u; c++) {
            if (((bits >> c) & 1u) != 0u)
                b.tc[s][c] = a.tc[s][c] + wrap_diff(b.tc[s][c], a.tc[s][c]);
        }
    }
}

/* --- Output ----------------------------------------------------------------- */

void
emit(V a, V flat_v, vec4 pos)
{
    /* TEXTUREPERSPECTIVE 0 with transformed vertices: the attributes times
       w, which the fragment shader multiplies by gl_FragCoord.w (1 / w,
       linear in screen space), so they come out linear in screen space. */
    float k = ((PROGRAM != 0u) && (vu.fb.z == 0.0)) ? pos.w : 1.0;

    gl_Position     = pos;
    gl_PointSize    = 1.0;
    v_diffuse       = a.diffuse * k;
    v_diffuse_flat  = flat_v.diffuse;
    v_specular      = a.specular * k;
    v_specular_flat = flat_v.specular;
    for (uint s = 0u; s < 8u; s++)
        v_tc[s] = a.tc[s] * k;
    v_fog   = a.fog * k;
    v_zr    = a.zr;
    v_line0 = ivec4(0);
    v_line1 = ivec4(0);
    clip_distances(a);
}

/* --- Culling (point fill and wireframe) --------------------------------------- */

/* Is the triangle not drawn: without area on the 1/16-pixel grid, or
   culled by its winding (spec/gpu-commands.md)? A transformed triangle with
   a vertex off the grid (behind the eye, or beyond the grid's edges) is
   judged by its homogeneous orientation. */
bool
culled(V a, V b, V c)
{
    int64_t det;

    if (a.on && b.on && c.on) {
        ivec2   ia   = snap16(a.scr);
        ivec2   ib   = snap16(b.scr);
        ivec2   ic   = snap16(c.scr);
        int64_t dx10 = int64_t(ib.x) - int64_t(ia.x);
        int64_t dy20 = int64_t(ic.y) - int64_t(ia.y);
        int64_t dx02 = int64_t(ia.x) - int64_t(ic.x);
        int64_t dy01 = int64_t(ia.y) - int64_t(ib.y);

        det = dx10 * dy20 - dx02 * dy01;
    } else {
        /* Screen y points down, clip y up: the screen winding is the
           opposite of the homogeneous determinant's. */
        float h = determinant(mat3(a.clip.xyw, b.clip.xyw, c.clip.xyw));

        det = (h < 0.0) ? int64_t(1) : (h > 0.0) ? int64_t(-1) : int64_t(0);
    }
    if (det == int64_t(0))
        return true;
    if ((vu.mode.z == 2u) && (det > int64_t(0)))
        return true;
    if ((vu.mode.z == 3u) && (det < int64_t(0)))
        return true;
    return false;
}

/* --- Lines ---------------------------------------------------------------- */

V
lerp_v(V a, V b, float t)
{
    V r = a;

    r.clip     = mix(a.clip, b.clip, t);
    r.world    = mix(a.world, b.world, t);
    r.diffuse  = mix(a.diffuse, b.diffuse, t);
    r.specular = mix(a.specular, b.specular, t);
    r.fog      = mix(a.fog, b.fog, t);
    for (uint s = 0u; s < 8u; s++)
        r.tc[s] = mix(a.tc[s], b.tc[s], t);
    project(r);
    return r;
}

/* How far inside the grid's edges a clipped line ends, in pixels: its
   endpoints stay on the grid whatever the rounding of the clip. */
#define GRID_EDGE 4194240.0

/* A transformed line clipped against the near and far planes and the user
   planes, linearly in clip space as the reference T&L clips, and against
   the 1/16-pixel grid's edges (far outside any target), so its endpoints
   have screen positions the grid holds. False when nothing is left. */
bool
clip_line(inout V a, inout V b)
{
    float t0 = 0.0;
    float t1 = 1.0;

    if (PROGRAM == 0u)
        return true;
    for (uint i = 0u; i < 12u; i++) {
        float d0;
        float d1;

        if (i == 0u) {
            d0 = a.clip.z;
            d1 = b.clip.z;
        } else if (i == 1u) {
            d0 = a.clip.w - a.clip.z;
            d1 = b.clip.w - b.clip.z;
        } else if (i < 6u) {
            /* The screen position within +-GRID_EDGE, as planes in clip
               space (w >= 0 once near and far hold): x_s = X0 + x/w * W/2,
               y_s = Y0 - y/w * H/2. */
            vec4 pl = (i == 2u) ? vec4(vu.vp.x, 0.0, 0.0, vu.vp.z + GRID_EDGE) :
                      (i == 3u) ? vec4(-vu.vp.x, 0.0, 0.0, GRID_EDGE - vu.vp.z) :
                      (i == 4u) ? vec4(0.0, -vu.vp.y, 0.0, vu.vp.w + GRID_EDGE) :
                                  vec4(0.0, vu.vp.y, 0.0, GRID_EDGE - vu.vp.w);

            d0 = dot(a.clip, pl);
            d1 = dot(b.clip, pl);
        } else {
            uint k = i - 6u;

            if (((vu.mode.x & VF_CLIPPING) == 0u) || (((vu.mode.w >> k) & 1u) == 0u))
                continue;
            d0 = plane_dist(a, k);
            d1 = plane_dist(b, k);
        }
        if ((d0 < 0.0) && (d1 < 0.0))
            return false;
        if (d0 < 0.0)
            t0 = max(t0, d0 / (d0 - d1));
        else if (d1 < 0.0)
            t1 = min(t1, d0 / (d0 - d1));
    }
    if (t0 >= t1)
        return false;
    {
        V na = (t0 > 0.0) ? lerp_v(a, b, t0) : a;
        V nb = (t1 < 1.0) ? lerp_v(a, b, t1) : b;

        a = na;
        b = nb;
    }
    return a.on && b.on;
}

/* The reference rasterizer's diamond test for a vertex at (fx, fy) 1/16
   pixels from its pixel's centre. */
bool
diamond(int fx, int fy, bool slope1, bool slope_pos1)
{
    int sum = abs(fx) + abs(fy);

    if (sum < 8)
        return true;
    if ((fx == (slope_pos1 ? -8 : 8)) && (fy == 0))
        return true;
    if ((fy == 8) && (fx == 0))
        return true;
    if (slope1 && (sum == 8)) {
        if (slope_pos1 && (fx < 0) && (fy > 0))
            return true;
        if (!slope_pos1 && (fx > 0) && (fy > 0))
            return true;
    }
    return false;
}

bool
dir_lt(float majext, int a, int b)
{
    return (majext > 0.0) ? (a < b) : (a > b);
}

/* One corner of the band for the line from a to b, its first vertex
   `flat_v` for flat shading: the band covers the pixels Direct3D's rule
   can pick on the major axis, two pixels wide, and the fragment shader
   keeps exactly those the rule picks (spec/gpu-commands.md, Lines). Its
   end pieces hold the endpoints' attributes, so attributes follow the
   major axis clamped to the segment. */
void
line_corner(V a, V b, V flat_v, bool lastpixel, uint corner)
{
    ivec2 i0 = snap16(a.scr);
    ivec2 i1 = snap16(b.scr);
    int   xs = i1.x - i0.x;
    int   ys = i1.y - i0.y;

    if ((xs == 0) && (ys == 0)) {
        kill();
        return;
    }
    bool    xmajor = abs(xs) >= abs(ys);
    float   majext = float(xmajor ? xs : ys) / 16.0;
    int64_t e0     = int64_t(xmajor ? ys : xs);
    int64_t e2     = int64_t(xmajor ? xs : ys);
    int64_t e1     = xmajor ? (int64_t(i0.y) * int64_t(i1.x) - int64_t(i1.y) * int64_t(i0.x))
                            : (int64_t(i0.x) * int64_t(i1.y) - int64_t(i1.x) * int64_t(i0.y));
    bool    slope1     = abs(xs) == abs(ys);
    bool    slope_pos1 = slope1 && ((e0 > int64_t(0)) == (e2 > int64_t(0)));
    int     px0        = (i0.x + (slope_pos1 ? 8 : 7)) >> 4;
    int     px1        = (i1.x + (slope_pos1 ? 8 : 7)) >> 4;
    int     py0        = (i0.y + 7) >> 4;
    int     py1        = (i1.y + 7) >> 4;
    bool    in0        = diamond(i0.x - px0 * 16, i0.y - py0 * 16, slope1, slope_pos1);
    bool    in1        = diamond(i1.x - px1 * 16, i1.y - py1 * 16, slope1, slope_pos1);
    int     step       = (majext > 0.0) ? 1 : -1;
    int     maj0       = xmajor ? i0.x : i0.y;
    int     maj1       = xmajor ? i1.x : i1.y;
    int     lmin       = xmajor ? px0 : py0;
    int     lmax       = xmajor ? px1 : py1;

    if (!(in0 || dir_lt(majext, maj0, lmin * 16)))
        lmin += step;
    if (!((!in1 && dir_lt(majext, lmax * 16, maj1)) || (in1 && lastpixel)))
        lmax -= step;
    if (dir_lt(majext, lmax, lmin)) {
        kill();
        return;
    }

    /* The band along the major axis: [first pixel's edge, a], [a, b], [b,
       last pixel's edge], the end pieces empty when the pixel lies inside
       the segment. */
    float fs    = float(step);
    float ua    = xmajor ? a.scr.x : a.scr.y;
    float ub    = xmajor ? b.scr.x : b.scr.y;
    float ufirst = float(lmin) - 0.5 * fs;
    float ulast  = float(lmax) + 0.5 * fs;
    uint  piece = corner / 6u;
    uint  k     = corner % 6u;
    /* Quad (u0, -1), (u1, -1), (u1, +1), (u0, +1) as triangles (0, 1, 2), (0, 2, 3). */
    uint  q     = (k < 3u) ? k : ((k == 3u) ? 0u : k - 2u);
    bool  far_u = (q == 1u) || (q == 2u);
    float side  = (q >= 2u) ? 1.0 : -1.0;
    float u0;
    float u1;
    V     ea;
    V     eb;

    if (piece == 0u) {
        u0 = ((ufirst - ua) * fs < 0.0) ? ufirst : ua;
        u1 = ua;
        ea = a;
        eb = a;
    } else if (piece == 1u) {
        u0 = ua;
        u1 = ub;
        ea = a;
        eb = b;
    } else {
        u0 = ub;
        u1 = ((ulast - ub) * fs > 0.0) ? ulast : ub;
        ea = b;
        eb = b;
    }

    /* The band's centre follows the snapped line. */
    float m0    = float(xmajor ? i0.y : i0.x) / 16.0;
    float slope = float(e0) / float(e2);
    float u     = far_u ? u1 : u0;
    float mc    = m0 + (u - float(maj0) / 16.0) * slope + side;
    vec2  s     = xmajor ? vec2(u, mc) : vec2(mc, u);
    V     e     = far_u ? eb : ea;
    float zn    = (PROGRAM == 0u) ? clamp(e.clip.z, 0.0, 1.0) : (e.clip.z / e.clip.w);

    emit(e, flat_v, to_clip(s, zn, e.w));
    for (uint i = 0u; i < 6u; i++)
        gl_ClipDistance[i] = 1.0; /* clipped already */
    v_line0 = ivec4(int(e0), int(e2), int(uint(uint64_t(e1) & 0xffffffffUL)), int(e1 >> 32));
    v_line1 = ivec4(lmin, lmax, (xmajor ? 1 : 0) | ((step < 0) ? 2 : 0), 0);
}

/* --- Points (command set 2.1) ---------------------------------------------- */

/* A point's size in pixels: the vertex's PSIZE or POINTSIZE, for a
   transformed vertex with POINTSCALEENABLE scaled by DirectX 8's formula
   on its distance from the eye and the viewport's height, then clamped to
   POINTSIZE_MIN and the largest size (spec/gpu-commands.md, Points). */
float
point_size(V a, int v)
{
    uint  n;
    /* A vertex program's oPts (POINTSIZE when it writes none), in pixels
       and never scaled (2.2). */
    float si = (PROGRAM == 2u) ? a.psize : (vu.slot[SLOT_PSIZE].z != 0u) ? fetch(SLOT_PSIZE, v, n).x : vu.pt.x;
    float s  = si;

    if (!(si > 0.0))
        si = 0.0;
    s = si;
    if ((PROGRAM == 1u) && ((vu.dx8.x & DX8_SCALE) != 0u)) {
        float d   = a.dist;
        float den = vu.ptscale.x + vu.ptscale.y * d + vu.ptscale.z * d * d;

        /* A tiny denominator gives a huge size, never 0 times infinity: a
           size of 0 stays 0 whatever the denominator. */
        s = (si == 0.0) ? 0.0 : (den > 0.0) ? vu.pt.w * si * inversesqrt(max(den, 1e-30)) : vu.pt.z;
        if (!(s >= 0.0))
            s = vu.pt.z; /* NaN */
    }
    /* As DirectX 8 states it: the largest first, then the smallest (which
       wins where they cross). */
    float r = (s > vu.pt.z) ? vu.pt.z : s;

    return (r < vu.pt.y) ? vu.pt.y : r;
}

/* Is a transformed point not drawn: its vertex outside the near or far
   plane, behind the eye, or outside an enabled user clip plane? A point
   inside them is drawn whole, clipped only by the scissor. */
bool
point_clipped(V a)
{
    if (PROGRAM == 0u)
        return false;
    if (!(a.clip.w > 0.0) || (a.clip.z < 0.0) || (a.clip.z > a.clip.w) || !a.on)
        return true;
    for (uint i = 0u; i < 6u; i++) {
        if (((vu.mode.x & VF_CLIPPING) != 0u) && (((vu.mode.w >> i) & 1u) != 0u) && (plane_dist(a, i) < 0.0))
            return true;
    }
    return false;
}

/* Corner `corner` (0-5: two triangles) of the square of side `size` centred
   on the point, at its depth and W, its colours everywhere; with
   POINTSPRITEENABLE every stage's first two coordinates span the square
   from (0, 0) at its top left to (1, 1) at its bottom right. */
void
point_corner(V a, V flat_v, float size, uint corner)
{
    /* Square (-, -), (+, -), (+, +), (-, +) as triangles (0, 1, 2), (0, 2, 3). */
    uint  q  = (corner < 3u) ? corner : ((corner == 3u) ? 0u : corner - 2u);
    vec2  uv = vec2(((q == 1u) || (q == 2u)) ? 1.0 : 0.0, (q >= 2u) ? 1.0 : 0.0);
    vec2  c  = snap(a.scr);
    vec2  s  = c + (uv - 0.5) * size;
    float zn = (PROGRAM == 0u) ? clamp(a.clip.z, 0.0, 1.0) : (a.clip.z / a.clip.w);

    if ((vu.dx8.x & DX8_SPRITE) != 0u) {
        for (uint i = 0u; i < 8u; i++)
            a.tc[i].xy = uv;
    }
    emit(a, flat_v, to_clip(s, zn, a.w));
    for (uint i = 0u; i < 6u; i++)
        gl_ClipDistance[i] = 1.0; /* clipped by its centre already */
}

/* The square of point a (vertex v); flat shading takes flat_v's colours:
   the point's own, or in point fill the triangle's first vertex's. */
void
point_shape(V a, V flat_v, int v, uint corner)
{
    float size = point_size(a, v);

    if (point_clipped(a) || !(size > 0.0)) {
        kill();
        return;
    }
    point_corner(a, flat_v, size, corner);
}

/* --- The shapes ----------------------------------------------------------- */

void
main()
{
    if (SHAPE == SHAPE_PLAIN) {
        V a = process(gl_VertexIndex);

        emit(a, a, place(a));
        return;
    }

    if (SHAPE == SHAPE_SPRITE) {
        ivec4 pr = pb.p[uint(gl_VertexIndex) / SPRITE_VERTICES];
        V     a  = process(pr.x);

        point_shape(a, a, pr.x, uint(gl_VertexIndex) % SPRITE_VERTICES);
        return;
    }

    if (SHAPE == SHAPE_FILLSPRITE) {
        /* Point fill of a triangle that is drawn: its three vertices'
           squares, culled as the triangle, WRAPn relative to its first
           vertex and flat shaded with it, as 2.0's point fill. */
        uint  per = 3u * SPRITE_VERTICES;
        ivec4 pr  = pb.p[uint(gl_VertexIndex) / per];
        uint  c   = uint(gl_VertexIndex) % per;
        V     t0  = process(pr.x);
        V     t1  = process(pr.y);
        V     t2  = process(pr.z);
        uint  k   = c / SPRITE_VERTICES;

        wrap(t0, t1);
        wrap(t0, t2);
        if (culled(t0, t1, t2)) {
            kill();
            return;
        }
        point_shape((k == 0u) ? t0 : (k == 1u) ? t1 : t2, t0, (k == 0u) ? pr.x : (k == 1u) ? pr.y : pr.z,
                    c % SPRITE_VERTICES);
        return;
    }

    if (SHAPE == SHAPE_LINE) {
        ivec4 pr = pb.p[uint(gl_VertexIndex) / BAND_VERTICES];
        V     a  = process(pr.x);
        V     b  = process(pr.y);
        V     f  = a;

        wrap(a, b);
        if (!clip_line(a, b)) {
            kill();
            return;
        }
        line_corner(a, b, f, pr.z != 0, uint(gl_VertexIndex) % BAND_VERTICES);
        return;
    }

    /* A triangle's shapes: its three vertices, WRAPn relative to the first. */
    {
        uint  per = (SHAPE == SHAPE_WIRE) ? (3u * BAND_VERTICES) : 3u;
        ivec4 pr  = pb.p[uint(gl_VertexIndex) / per];
        uint  c   = uint(gl_VertexIndex) % per;
        V     t[3];

        t[0] = process(pr.x);
        t[1] = process(pr.y);
        t[2] = process(pr.z);
        wrap(t[0], t[1]);
        wrap(t[0], t[2]);

        if (SHAPE == SHAPE_WRAP) {
            emit(t[c], t[0], place(t[c]));
            return;
        }
        if (culled(t[0], t[1], t[2])) {
            kill();
            return;
        }
        if (SHAPE == SHAPE_POINTS) {
            emit(t[c], t[0], place(t[c]));
            return;
        }
        /* SHAPE_WIRE: edge e (first-second, second-third, third-first) when
           its flag is set, a line with the triangle's first vertex for
           flat shading and LASTPIXEL from the state. */
        {
            uint e  = c / BAND_VERTICES;
            V    a  = t[e];
            V    b  = t[(e + 1u) % 3u];

            if (((uint(pr.w) >> e) & 1u) == 0u) {
                kill();
                return;
            }
            if (!clip_line(a, b)) {
                kill();
                return;
            }
            line_corner(a, b, t[0], (vu.mode.x & VF_LASTPIXEL) != 0u, c % BAND_VERTICES);
        }
    }
}
