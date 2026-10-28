/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the fragment shader, Direct3D 7's fixed-function
 *          pixel pipeline as the project's spec/gpu-commands.md defines
 *          it, following Microsoft's reference rasterizer where the
 *          specification leaves details open: up to eight texture
 *          stages with every operation and argument, bump mapping,
 *          colour keys, specular, vertex and table fog, the alpha test,
 *          dithering; command set 2.1's DirectX 8 additions (the
 *          operations MULTIPLYADD and LERP with their third argument, the
 *          TEMP register and RESULTARG, volume textures); with
 *          COMPUTED_DEPTH, the depth value too (W
 *          buffering, Z on the guest buffer's grid in a 32-bit float
 *          image, ZBIAS where the host's bias cannot give it), ZRANGE
 *          applied here as the viewport's range is 0-1; command set
 *          2.2's pixel programs (PIXEL_PROGRAM), run here in place of the
 *          stages. Blending,
 *          the depth and stencil tests are the host GPU's fixed
 *          function, set per pipeline.
 *
 *          The layout of the uniform block is mirrored by the renderer
 *          (vid_vgpu98_r3d.c, r3d_ubo_t).
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

/* Lines and wireframe edges: keep only the pixels Direct3D's line rule
   picks (the vertex shader's band covers them). */
layout(constant_id = 2) const bool LINE_RULE = false;
/* The vertex program (the vertex shader's PROGRAM): 0 pre-transformed
   vertices, whose screen z and rhw come in v_zr; otherwise transformed ones,
   whose come from the position (gl_FragCoord), which the host GPU's
   clipper keeps exact. */
layout(constant_id = 1) const uint PROGRAM = 0u;
/* 2.2: a pixel program runs in place of the texture stages. */
layout(constant_id = 3) const bool PIXEL_PROGRAM = false;

/* Texture operations (VGPU98_TOP_*) */
#define TOP_DISABLE                   1u
#define TOP_SELECTARG1                2u
#define TOP_SELECTARG2                3u
#define TOP_MODULATE                  4u
#define TOP_MODULATE2X                5u
#define TOP_MODULATE4X                6u
#define TOP_ADD                       7u
#define TOP_ADDSIGNED                 8u
#define TOP_ADDSIGNED2X               9u
#define TOP_SUBTRACT                  10u
#define TOP_ADDSMOOTH                 11u
#define TOP_BLENDDIFFUSEALPHA         12u
#define TOP_BLENDTEXTUREALPHA         13u
#define TOP_BLENDFACTORALPHA          14u
#define TOP_BLENDTEXTUREALPHAPM       15u
#define TOP_BLENDCURRENTALPHA         16u
#define TOP_PREMODULATE               17u
#define TOP_MODULATEALPHA_ADDCOLOR    18u
#define TOP_MODULATECOLOR_ADDALPHA    19u
#define TOP_MODULATEINVALPHA_ADDCOLOR 20u
#define TOP_MODULATEINVCOLOR_ADDALPHA 21u
#define TOP_BUMPENVMAP                22u
#define TOP_BUMPENVMAPLUMINANCE       23u
#define TOP_DOTPRODUCT3               24u
#define TOP_MULTIPLYADD               25u /* 2.1 */
#define TOP_LERP                      26u /* 2.1 */

/* Argument selectors (VGPU98_TA_*) */
#define TA_TEMP 5u /* 2.1: the temporary register */

/* Stage flags (ops2.w) */
#define SF_BOUND    1u /* a texture is bound */
#define SF_KEYED    2u /* the texture has a colour key: u_key holds its matches */
#define SF_BORDER_U 4u /* BORDER addressing done here (the host has no custom border colours) */
#define SF_BORDER_V 8u
#define SF_CUBE     16u /* a cube map: u_cube (and u_cube_key) */
#define SF_VOLUME   64u /* 2.1: a volume texture: u_vol */
#define SF_BORDER_W 128u /* 2.1: BORDER on a volume's third coordinate done here */

/* mode.z flags */
#define MF_SPECULAR 1u
#define MF_COLORKEY 2u
#define MF_FLAT     4u
#define MF_DITHER   8u
#define MF_LINEAR   16u /* TEXTUREPERSPECTIVE 0: transformed vertices' attributes come times w */

struct Stage {
    uvec4 ops;    /* colorop, colorarg1, colorarg2, alphaop */
    uvec4 ops2;   /* alphaarg1, alphaarg2, TEXTURETRANSFORMFLAGS, flags */
    vec4  bump;   /* BUMPENVMAT00, 01, 10, 11 */
    vec4  misc;   /* BUMPENVLSCALE, BUMPENVLOFFSET, 2^MIPMAPLODBIAS, unused */
    vec4  border; /* BORDERCOLOR, for SF_BORDER_U/V/W */
    uvec4 ops3;   /* 2.1: colorarg0, alphaarg0, resultarg (1 current, 5 temp), unused */
};

layout(std140, set = 0, binding = 0) uniform State {
    Stage st[8];
    vec4  tfactor;
    vec4  fogcolor;
    vec4  fog;    /* start, end, density, unused */
    uvec4 mode;   /* x fog, y alpha test, z flags, w depth (below) */
    vec4  wrange; /* wnear, |wfar - wnear| (0: infinite or empty), wnear, 1 / (wfar - wnear) or 1 */
    vec4  dither; /* the target's levels per channel: 2^bits - 1 for r, g, b */
    vec4  zrange; /* minz, maxz - minz: ZRANGE, which a computed depth's viewport leaves out */
    /* Command set 2.2 */
    uvec4 ph;     /* the pixel program's instructions, the stages it samples, its flags */
    vec4  prange; /* x: its registers' range */
    vec4  pc[8];  /* the pixel constants */
} u;

/* 2.2: the pixel program's instructions. */
layout(std430, set = 1, binding = 12) readonly buffer PProg {
    uvec4 ins[];
} pprog;

layout(set = 0, binding = 1) uniform sampler2D u_tex[8];
layout(set = 0, binding = 2) uniform sampler2D u_key[8];
layout(set = 0, binding = 6) uniform samplerCube u_cube[8];
layout(set = 0, binding = 7) uniform samplerCube u_cube_key[8];
layout(set = 0, binding = 9) uniform sampler3D u_vol[8]; /* 2.1: volume textures */

layout(location = 0) in vec4 v_diffuse;
layout(location = 1) flat in vec4 v_diffuse_flat;
layout(location = 2) in vec4 v_specular;
layout(location = 3) flat in vec4 v_specular_flat;
layout(location = 4) in vec4 v_tc[8]; /* each stage's coordinates */
layout(location = 12) noperspective in vec2 v_zr;
layout(location = 13) flat in ivec4 v_line0; /* e0, e2, e1 low, e1 high */
layout(location = 14) flat in ivec4 v_line1; /* first and last pixel on the major axis, x-major | step < 0 << 1 */
layout(location = 15) in float v_fog;        /* the fog factor (2.2: apart from the colours) */

layout(location = 0) out vec4 o_color;

/* What the smooth attributes are multiplied by: gl_FragCoord.w for
   transformed vertices with TEXTUREPERSPECTIVE 0 (the vertex shader sent
   them times w), which makes them linear in screen space; else 1. */
float lin = 1.0;

/* A stage's texture coordinates: with TEXTURETRANSFORMFLAGS' PROJECTED and
   a COUNT of 2-4, divided by component COUNT - 1. A 2D texture takes the
   first two, a cube map the first three (a direction). */
vec4
stage_tc(uint s)
{
    uint f     = u.st[s].ops2.z;
    uint count = f & 0xffu;
    vec4 tc    = v_tc[s] * lin;

    if (((f & 0x100u) != 0u) && (count >= 2u))
        return tc / tc[count - 1u];
    return tc;
}

/* The temporary register (2.1): (0, 0, 0, 0) until a stage writes it. */
vec4 temp = vec4(0.0);

/* The pixel is not drawn: a colour key, a line's rule, a program's kill. */
bool killed = false;

/* A stage argument: the selector, then the complement, then alpha replicated
   (the reference rasterizer's order). An alpha argument complements only
   alpha. */
vec4
stage_arg(uint a, bool alpha_only, vec4 diffuse, vec4 current, vec4 texel, vec4 specular)
{
    uint sel = a & 15u;
    vec4 v   = (sel == 0u) ? diffuse : (sel == 1u) ? current : (sel == 2u) ? texel : (sel == 3u) ? u.tfactor :
               (sel == TA_TEMP) ? temp : specular;

    if ((a & 0x10u) != 0u) {
        if (alpha_only)
            v.a = 1.0 - v.a;
        else
            v = vec4(1.0) - v;
    }
    if (!alpha_only && ((a & 0x20u) != 0u))
        v.rgb = vec3(v.a);
    return v;
}

/* One operation on colour (vec3) or alpha (in .x), with the blend factors
   it may need, and 2.1's third argument a0. */
vec3
stage_op(uint op, vec3 a0, vec3 a1, vec3 a2, float a1alpha, float diff_a, float tex_a, float cur_a)
{
    switch (op) {
        case TOP_MULTIPLYADD:
            return a0 + a1 * a2;
        case TOP_LERP:
            return a0 * a1 + (vec3(1.0) - a0) * a2;
        case TOP_SELECTARG1:
        case TOP_PREMODULATE:
            return a1;
        case TOP_SELECTARG2:
            return a2;
        case TOP_MODULATE:
            return a1 * a2;
        case TOP_MODULATE2X:
            return a1 * a2 * 2.0;
        case TOP_MODULATE4X:
            return a1 * a2 * 4.0;
        case TOP_ADD:
            return a1 + a2;
        case TOP_ADDSIGNED:
            return a1 + a2 - 0.5;
        case TOP_ADDSIGNED2X:
            return (a1 + a2 - 0.5) * 2.0;
        case TOP_SUBTRACT:
            return a1 - a2;
        case TOP_ADDSMOOTH:
            return a1 + (1.0 - a1) * a2;
        case TOP_BLENDDIFFUSEALPHA:
            return mix(a2, a1, diff_a);
        case TOP_BLENDTEXTUREALPHA:
            return mix(a2, a1, tex_a);
        case TOP_BLENDFACTORALPHA:
            return mix(a2, a1, u.tfactor.a);
        case TOP_BLENDCURRENTALPHA:
            return mix(a2, a1, cur_a);
        case TOP_BLENDTEXTUREALPHAPM:
            return a1 + (1.0 - tex_a) * a2;
        case TOP_MODULATEALPHA_ADDCOLOR:
            return a1 + a1alpha * a2;
        case TOP_MODULATECOLOR_ADDALPHA:
            return a1 * a2 + a1alpha;
        case TOP_MODULATEINVALPHA_ADDCOLOR:
            return (1.0 - a1alpha) * a2 + a1;
        case TOP_MODULATEINVCOLOR_ADDALPHA:
            return (1.0 - a1) * a2 + a1alpha;
        default:
            return a1;
    }
}

/* --- Pixel programs (command set 2.2) ---------------------------------------- */

/* Instruction fields and files (vid_vgpu98_prog.h, spec/gpu-commands.md). */
#define PF_TEMP   1u
#define PF_INPUT  2u
#define PF_CONST  3u
#define PF_COORD  5u
#define PF_STAGE  7u
#define PF_FIXED  8u
#define INS_SAT     0x01000000u
#define INS_PROJECT 0x10000000u
#define INS_NOCLAMP 0x20000000u

#define PI_MOV    0x01u
#define PI_ADD    0x02u
#define PI_SUB    0x03u
#define PI_MUL    0x04u
#define PI_MAD    0x05u
#define PI_DP3    0x06u
#define PI_DP4    0x07u
#define PI_LRP    0x08u
#define PI_CND    0x09u
#define PI_CMP    0x0au
#define PI_MIN    0x0bu
#define PI_MAX    0x0cu
#define PI_SLT    0x0du
#define PI_SGE    0x0eu
#define PI_RCP    0x0fu
#define PI_RSQ    0x10u
#define PI_EXP    0x11u
#define PI_LOG    0x12u
#define PI_EXPP   0x13u
#define PI_LOGP   0x14u
#define PI_FRC    0x15u
#define PI_LIT    0x16u
#define PI_DST    0x17u
#define PI_SAMPLE 0x20u
#define PI_BEM    0x21u
#define PI_LUM    0x22u
#define PI_RFL    0x23u
#define PI_KILL   0x24u
#define PI_DEPTH  0x25u

#define FLT_MAX_P 3.402823466e+38
#define INF_P     uintBitsToFloat(0x7f800000u)

vec4  pp_r[16];        /* temporaries */
vec4  pp_v[2];         /* the colours */
bool  pp_depth = false; /* a DEPTH ran ... */
float pp_z     = 0.0;   /* ... with this depth */

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

/* A source operand of a pixel program. COORD n is stage n's coordinates as
   interpolated, before any projection. */
vec4
pp_src(uint s)
{
    uint file = s & 15u;
    uint reg  = (s >> 4u) & 255u;
    vec4 x;

    if (file == PF_TEMP)
        x = pp_r[reg & 15u];
    else if (file == PF_INPUT)
        x = pp_v[reg & 1u];
    else if (file == PF_CONST)
        x = u.pc[reg & 7u];
    else if (file == PF_COORD)
        x = v_tc[reg & 7u] * lin;
    else
        x = fixed_vec(reg);
    return apply_mod(swizzle(x, (s >> 12u) & 255u), (s >> 20u) & 15u);
}

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

/* Stage s's texture sampled at coordinates tc (2D: x, y; a cube map or a
   volume: x, y, z), with the gradients of the coordinates as computed, the
   stage's border and colour key as the fixed function applies them; a stage
   without a texture reads (0, 0, 0, 0). */
vec4
pp_sample(uint s, vec4 tc)
{
    Stage stg = u.st[s];
    vec3  gx  = dFdx(tc.xyz) * stg.misc.z;
    vec3  gy  = dFdy(tc.xyz) * stg.misc.z;
    vec4  t;

    if ((stg.ops2.w & SF_BOUND) == 0u)
        return vec4(0.0);
    if ((stg.ops2.w & SF_VOLUME) != 0u) {
        t = textureGrad(u_vol[s], tc.xyz, gx, gy);
        if ((((stg.ops2.w & SF_BORDER_U) != 0u) && ((tc.x < 0.0) || (tc.x > 1.0))) ||
            (((stg.ops2.w & SF_BORDER_V) != 0u) && ((tc.y < 0.0) || (tc.y > 1.0))) ||
            (((stg.ops2.w & SF_BORDER_W) != 0u) && ((tc.z < 0.0) || (tc.z > 1.0))))
            t = stg.border;
        return t;
    }
    if ((stg.ops2.w & SF_CUBE) != 0u) {
        t = textureGrad(u_cube[s], tc.xyz, gx, gy);
        if (((u.mode.z & MF_COLORKEY) != 0u) && ((stg.ops2.w & SF_KEYED) != 0u) &&
            (textureGrad(u_cube_key[s], tc.xyz, gx, gy).r > 0.0))
            killed = true;
        return t;
    }
    t = textureGrad(u_tex[s], tc.xy, gx.xy, gy.xy);
    if ((((stg.ops2.w & SF_BORDER_U) != 0u) && ((tc.x < 0.0) || (tc.x > 1.0))) ||
        (((stg.ops2.w & SF_BORDER_V) != 0u) && ((tc.y < 0.0) || (tc.y > 1.0))))
        t = stg.border;
    if (((u.mode.z & MF_COLORKEY) != 0u) && ((stg.ops2.w & SF_KEYED) != 0u) &&
        (textureGrad(u_key[s], tc.xy, gx.xy, gy.xy).r > 0.0))
        killed = true;
    return t;
}

/* PROJECT (SAMPLE, BEM): coordinates divided as stage s's
   TEXTURETRANSFORMFLAGS say, by component COUNT - 1 with PROJECTED and a
   COUNT of 2-4. */
vec4
pp_project(uint s, vec4 a)
{
    uint f     = u.st[s].ops2.z;
    uint count = f & 0xffu;

    if (((f & 0x100u) != 0u) && (count >= 2u) && (count <= 4u))
        return a / a[count - 1u];
    return a;
}

/* The pixel program: its colour (TEMP 0, clamped to 0-1). */
vec4
run_pixel_program(vec4 diffuse, vec4 specular)
{
    float range = u.prange.x;

    for (uint i = 0u; i < 16u; i++)
        pp_r[i] = vec4(0.0);
    pp_v[0] = clamp(diffuse, 0.0, 1.0);
    pp_v[1] = clamp(specular, 0.0, 1.0);
    for (uint i = 0u; i < u.ph.x; i++) {
        uvec4 in_ = pprog.ins[i];
        uint  op  = in_.x & 255u;
        uint  dr  = (in_.x >> 12u) & 15u;
        uint  m   = (in_.x >> 20u) & 15u;
        int   sh  = int((in_.x >> 25u) & 7u) - ((((in_.x >> 25u) & 4u) != 0u) ? 8 : 0);
        bvec4 wm  = bvec4((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u, (m & 8u) != 0u);
        vec4  a   = pp_src(in_.y);
        vec4  r;

        if (op == PI_SAMPLE) {
            uint st = ((in_.z >> 4u) & 255u) & 7u;

            if ((in_.x & INS_PROJECT) != 0u)
                a = pp_project(st, a);
            r = pp_sample(st, a);
        } else if (op == PI_BEM) {
            uint st = ((in_.w >> 4u) & 255u) & 7u;
            vec4 b  = pp_src(in_.z);
            vec4 mt = u.st[st].bump;

            if ((in_.x & INS_PROJECT) != 0u)
                a = pp_project(st, a);
            r = vec4(a.x + mt.x * b.x + mt.z * b.y, a.y + mt.y * b.x + mt.w * b.y, a.z, a.w);
        } else if (op == PI_LUM) {
            vec4  b   = pp_src(in_.z);
            Stage stg = u.st[((in_.w >> 4u) & 255u) & 7u];

            r = a * clamp(b.z * stg.misc.x + stg.misc.y, 0.0, 1.0);
        } else if (op == PI_RFL) {
            vec3  b  = pp_src(in_.z).xyz;
            float nn = dot(a.xyz, a.xyz);

            r = vec4((nn == 0.0) ? -b : 2.0 * dot(a.xyz, b) / nn * a.xyz - b, 0.0);
        } else if (op == PI_KILL) {
            if (any(lessThan(a.xyz, vec3(0.0))))
                killed = true;
            continue;
        } else if (op == PI_DEPTH) {
            pp_depth = true;
            pp_z     = clamp((a.y == 0.0) ? 1.0 : a.x / a.y, 0.0, 1.0);
            continue;
        } else
            r = op_eval(op, a, pp_src(in_.z), pp_src(in_.w));
        if (sh != 0)
            r *= exp2(float(sh));
        if ((in_.x & INS_SAT) != 0u)
            r = clamp(r, 0.0, 1.0);
        if ((in_.x & INS_NOCLAMP) == 0u)
            r = clamp(r, -range, range);
        pp_r[dr] = mix(pp_r[dr], r, wm);
    }
    return clamp(pp_r[0], 0.0, 1.0);
}

const float bayer[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);

/* The reference rasterizer's rounding: ceil(a / b - 1/2), to nearest with
   halves down. */
int64_t
div_round_down(int64_t a, int64_t b)
{
    int     neg = 0;
    int64_t q;

    if (a < int64_t(0)) {
        neg++;
        a = -a;
    }
    if (b < int64_t(0)) {
        neg++;
        b = -b;
    }
    a += (b - ((neg == 1) ? int64_t(0) : int64_t(1))) >> 1;
    q = a / b;
    return (neg == 1) ? -q : q;
}

/* Is this pixel one of the line's, by Direct3D's rule: on the major axis
   between the first and last pixel, and on the minor axis where the line
   rounds to (spec/gpu-commands.md)? */
bool
line_pixel()
{
    ivec2   p      = ivec2(floor(gl_FragCoord.xy));
    bool    xmajor = (v_line1.z & 1) != 0;
    bool    back   = (v_line1.z & 2) != 0;
    int     m      = xmajor ? p.x : p.y;
    int     n      = xmajor ? p.y : p.x;
    int64_t e1     = (int64_t(v_line0.w) << 32) | int64_t(uint(v_line0.z));

    if (back ? ((m > v_line1.x) || (m < v_line1.y)) : ((m < v_line1.x) || (m > v_line1.y)))
        return false;
    return int64_t(n) == div_round_down(int64_t(v_line0.x) * int64_t(m * 16) + e1, int64_t(v_line0.y) * int64_t(16));
}

void
main()
{
    bool  flat_shade = (u.mode.z & MF_FLAT) != 0u;
    float k          = ((PROGRAM != 0u) && ((u.mode.z & MF_LINEAR) != 0u)) ? gl_FragCoord.w : 1.0;
    vec4  diffuse    = flat_shade ? v_diffuse_flat : v_diffuse * k;
    vec4  specular   = flat_shade ? v_specular_flat : v_specular * k;
    float fog_f      = v_fog * k;
    /* The specular argument: its colour, alpha 0 (the reference rasterizer
       keeps no specular alpha per pixel). */
    vec4  spec_arg   = vec4(specular.rgb, 0.0);
#ifdef COMPUTED_DEPTH
    /* The viewport's depth range is 0-1 here: ZRANGE goes on z now. */
    vec2  zr         = (PROGRAM == 0u) ? v_zr : vec2(u.zrange.x + gl_FragCoord.z * u.zrange.y, gl_FragCoord.w);
#else
    vec2  zr         = (PROGRAM == 0u) ? v_zr : vec2(gl_FragCoord.z, gl_FragCoord.w);
#endif
    vec4  current    = diffuse;
    vec2  bump_off   = vec2(0.0);
    float bump_l     = 1.0;
    bool  prev_bump  = false;
    bool  prev_premod_c = false;
    bool  prev_premod_a = false;
    vec3  dx[8];
    vec3  dy[8];
    uint  s;

    lin = k;
    /* Off the line: discarded with the colour key, after the derivatives
       (a quad's helper pixels keep them defined). */
    killed = LINE_RULE && !line_pixel();

    /* Derivatives of every coordinate set, taken where control flow is
       uniform: a bumped stage samples with its unbumped coordinates' slopes. */
    for (s = 0u; s < 8u; s++) {
        dx[s] = dFdx(stage_tc(s).xyz);
        dy[s] = dFdy(stage_tc(s).xyz);
    }

    if (PIXEL_PROGRAM)
        current = run_pixel_program(diffuse, specular);
    for (s = 0u; !PIXEL_PROGRAM && (s < 8u); s++) {
        Stage stg = u.st[s];
        uint  cop = stg.ops.x;
        uint  aop = stg.ops.w;
        vec4  texel = vec4(0.0); /* a stage without a texture reads transparent black */
        vec4  next;

        if (cop == TOP_DISABLE)
            break;

        if ((stg.ops2.w & SF_BOUND) != 0u) {
            if ((stg.ops2.w & SF_VOLUME) != 0u) {
                /* A volume (2.1): three coordinates, BORDER on each where
                   the host has no custom border colours; no colour keys
                   or bump offsets. */
                vec3 uvw = stage_tc(s).xyz;

                texel = textureGrad(u_vol[s], uvw, dx[s] * stg.misc.z, dy[s] * stg.misc.z);
                if ((((stg.ops2.w & SF_BORDER_U) != 0u) && ((uvw.x < 0.0) || (uvw.x > 1.0))) ||
                    (((stg.ops2.w & SF_BORDER_V) != 0u) && ((uvw.y < 0.0) || (uvw.y > 1.0))) ||
                    (((stg.ops2.w & SF_BORDER_W) != 0u) && ((uvw.z < 0.0) || (uvw.z > 1.0))))
                    texel = stg.border;
            } else if ((stg.ops2.w & SF_CUBE) != 0u) {
                /* A direction from the cube's centre; bump offsets and border
                   addressing are 2D textures' only. */
                vec3 dir = stage_tc(s).xyz;
                vec3 gx  = dx[s] * stg.misc.z;
                vec3 gy  = dy[s] * stg.misc.z;

                texel = textureGrad(u_cube[s], dir, gx, gy);
                if (((u.mode.z & MF_COLORKEY) != 0u) && ((stg.ops2.w & SF_KEYED) != 0u) &&
                    (textureGrad(u_cube_key[s], dir, gx, gy).r > 0.0))
                    killed = true;
            } else {
                vec2 uv = stage_tc(s).xy + (prev_bump ? bump_off : vec2(0.0));
                vec2 gx = dx[s].xy * stg.misc.z;
                vec2 gy = dy[s].xy * stg.misc.z;

                texel = textureGrad(u_tex[s], uv, gx, gy);
                if ((((stg.ops2.w & SF_BORDER_U) != 0u) && ((uv.x < 0.0) || (uv.x > 1.0))) ||
                    (((stg.ops2.w & SF_BORDER_V) != 0u) && ((uv.y < 0.0) || (uv.y > 1.0))))
                    texel = stg.border;
                if (((u.mode.z & MF_COLORKEY) != 0u) && ((stg.ops2.w & SF_KEYED) != 0u) &&
                    (textureGrad(u_key[s], uv, gx, gy).r > 0.0))
                    killed = true;
            }
        }

        if ((cop == TOP_BUMPENVMAP) || (cop == TOP_BUMPENVMAPLUMINANCE)) {
            /* No blending here: the texel (du, dv, l) offsets the next
               stage's coordinates and may scale its texture. */
            if ((stg.ops2.w & SF_BOUND) != 0u) {
                bump_off  = vec2(stg.bump.x * texel.r + stg.bump.z * texel.g, stg.bump.y * texel.r + stg.bump.w * texel.g);
                bump_l    = (cop == TOP_BUMPENVMAPLUMINANCE) ? clamp(texel.b * stg.misc.x + stg.misc.y, 0.0, 1.0) : 1.0;
                prev_bump = true;
            }
            prev_premod_c = false;
            prev_premod_a = false;
            continue;
        }
        if (prev_bump)
            texel.rgb *= bump_l;
        prev_bump = false;

        /* PREMODULATE in the previous stage: its result times this stage's texture. */
        if (prev_premod_c)
            current.rgb *= texel.rgb;
        if (prev_premod_a)
            current.a *= texel.a;

        vec4 c1 = stage_arg(stg.ops.y, false, diffuse, current, texel, spec_arg);
        vec4 c2 = stage_arg(stg.ops.z, false, diffuse, current, texel, spec_arg);
        /* 2.1's third argument, for MULTIPLYADD and LERP only. */
        vec4 c0 = ((cop == TOP_MULTIPLYADD) || (cop == TOP_LERP)) ?
                      stage_arg(stg.ops3.x, false, diffuse, current, texel, spec_arg) : vec4(0.0);

        if (cop == TOP_DOTPRODUCT3) {
            float d = 4.0 * dot(c1.rgb - 0.5, c2.rgb - 0.5);
            next    = vec4(d);
        } else {
            next.rgb = stage_op(cop, c0.rgb, c1.rgb, c2.rgb, c1.a, diffuse.a, texel.a, current.a);
            if (aop == TOP_DISABLE)
                next.a = current.a;
            else {
                vec4 x1 = stage_arg(stg.ops2.x, true, diffuse, current, texel, spec_arg);
                vec4 x2 = stage_arg(stg.ops2.y, true, diffuse, current, texel, spec_arg);
                vec4 x0 = ((aop == TOP_MULTIPLYADD) || (aop == TOP_LERP)) ?
                              stage_arg(stg.ops3.y, true, diffuse, current, texel, spec_arg) : vec4(0.0);

                next.a = stage_op(aop, vec3(x0.a), vec3(x1.a), vec3(x2.a), x1.a, diffuse.a, texel.a, current.a).x;
            }
        }
        /* RESULTARG (2.1): the result into the temporary register, current
           as it was; otherwise into current. */
        if (stg.ops3.z == TA_TEMP)
            temp = clamp(next, 0.0, 1.0);
        else
            current = clamp(next, 0.0, 1.0);
        prev_premod_c = (cop == TOP_PREMODULATE);
        prev_premod_a = (aop == TOP_PREMODULATE);
    }

    /* A texel matching its texture's colour key, in any stage, with a
       non-zero weight in the filter, removes the pixel. */
    if (killed)
        discard;

    /* Alpha test, on alpha as an 8-bit value. */
    uint afunc = (u.mode.y >> 8u) & 15u;
    if (afunc != 0u) {
        uint a   = uint(clamp(current.a, 0.0, 1.0) * 255.0 + 0.5);
        uint ref = u.mode.y & 255u;
        bool pass;

        switch (afunc) {
            case 1u: pass = false; break;
            case 2u: pass = a < ref; break;
            case 3u: pass = a == ref; break;
            case 4u: pass = a <= ref; break;
            case 5u: pass = a > ref; break;
            case 6u: pass = a != ref; break;
            case 7u: pass = a >= ref; break;
            default: pass = true; break;
        }
        if (!pass)
            discard;
    }

    /* With a pixel program the specular add is the program's own (2.2). */
    if (!PIXEL_PROGRAM && ((u.mode.z & MF_SPECULAR) != 0u))
        current.rgb = min(current.rgb + specular.rgb, vec3(1.0));

    /* Fog: the vertex factor (specular alpha), or the table's from W or, with
       an affine projection (WRANGE 1, 1), from Z. */
    if ((u.mode.x & 0x10u) != 0u) {
        uint  table = u.mode.x & 3u;
        float f;

        if (table == 0u)
            f = fog_f;
        else {
            float index = ((u.mode.x & 0x20u) != 0u) ? ((zr.y > 0.0) ? (1.0 / zr.y) : 0.0) : clamp(zr.x, 0.0, 1.0);

            if (table == 1u)
                f = exp(-(u.fog.z * index));
            else if (table == 2u)
                f = exp(-((u.fog.z * index) * (u.fog.z * index)));
            else if (index >= u.fog.y)
                f = 0.0;
            else if (index <= u.fog.x)
                f = 1.0;
            else
                f = (u.fog.y - index) / (u.fog.y - u.fog.x);
        }
        f           = clamp(f, 0.0, 1.0);
        current.rgb = f * current.rgb + (1.0 - f) * u.fogcolor.rgb;
    }

    /* A 4x4 ordered dither of half a step of the target's precision. */
    if ((u.mode.z & MF_DITHER) != 0u) {
        uvec2 p = uvec2(gl_FragCoord.xy) & 3u;
        float t = (bayer[p.y * 4u + p.x] + 0.5) / 16.0 - 0.5;

        current.rgb = clamp(current.rgb + t / u.dither.rgb, 0.0, 1.0);
    }

    o_color = current;

#ifdef COMPUTED_DEPTH
    /* mode.w: ZENABLE (bits 1:0); 4, Z on the guest buffer's grid in a
       float image; 8, W itself in a float image; ZBIAS (15:8); the guest
       buffer's bits (23:16). The renderer's depth_in and depth_out read
       the same values. */
    {
        uint  zmode = u.mode.w & 3u;
        int   bits  = int((u.mode.w >> 16u) & 255u);
        float top   = ldexp(1.0, bits) - 1.0; /* the guest buffer's largest step, exactly */
        float n     = float((u.mode.w >> 8u) & 255u);
        float d;

        if ((zmode == 2u) && ((u.mode.w & 8u) != 0u)) {
            /* W itself, as W * 2^-64: ordered and exact whatever the W
               range. ZBIAS moves it n steps of the guest buffer toward the
               viewer, in W units, or for a range with no span by n / 65535
               of W (a 16-bit step at least: a 24-bit one relative to W is
               one ulp, lost in the rasterizer's rounding). An RHW of 0 is
               W at infinity, a negative one the near end, as the
               reference's clamp gives. */
            if (zr.y > 0.0) {
                float w = 1.0 / zr.y;

                w = (u.wrange.y > 0.0) ? w - (n / top) * u.wrange.y : w * (1.0 - n / 65535.0);
                d = min(ldexp(max(w, 0.0), -64), 1.0);
            } else
                d = (zr.y == 0.0) ? 1.0 : 0.0;
        } else if (PIXEL_PROGRAM && pp_depth && (zmode != 2u)) {
            /* A pixel program's DEPTH (2.2): the buffer's value as it is, no
               ZRANGE or ZBIAS, on a fixed-point buffer's grid. */
            d = pp_z;
            if ((u.mode.w & 4u) != 0u)
                d = ldexp(roundEven(d * top), -bits);
        } else {
            if (zmode == 2u) /* the reference's form, in a fixed-point image: RHW 0 is the far end */
                d = (zr.y > 0.0) ? (min(1.0 / zr.y, 3.0e38) - u.wrange.z) * u.wrange.w : ((zr.y == 0.0) ? 1.0 : 0.0);
            else
                d = zr.x;
            d = clamp(d, 0.0, 1.0);
            if ((u.mode.w & 4u) != 0u) {
                /* The grid held exactly: step k as k / 2^bits. ZBIAS takes
                   n whole steps toward the viewer. */
                float k = max(roundEven(d * top) - n, 0.0);

                d = ldexp(k, -bits);
            } else
                d = max(d - n / top, 0.0); /* ZBIAS: n steps of the buffer's resolution */
        }
        gl_FragDepth = d;
    }
#endif
}
