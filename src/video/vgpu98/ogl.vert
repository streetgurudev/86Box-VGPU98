/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the OpenGL device's vertices.
 *          GL 1.1's per-vertex
 *          operations as the specification defines them: the modelview and
 *          projection matrices, normals through the modelview's inverse
 *          transpose, lighting by GL's equation (attenuation, spotlights,
 *          local viewer, two sides, colour material), texture coordinate
 *          generation and the texture matrix, the user's clip planes and
 *          the fog distance. The context's registers are the uniform block
 *          (vid_vgpu98_oglcmd.h, VGPU98_OGL_R_*).
 *
 *          The vertices and the 16-bit indices are read from storage
 *          buffers. Triangles are drawn as they are; points and lines as
 *          quads made here in window coordinates, six vertices each, so
 *          GL's rules hold: aliased points are squares around snapped
 *          centres, aliased wide lines columns or rows of fragments, smooth
 *          ones have their coverage computed in ogl.frag. Vulkan's viewport
 *          is GL's, upside down (GL's rows count up from the bottom).
 *
 *          With DRAW_CLIP the vertices are the ICD's own: clip coordinates,
 *          clipped, lit, their fog distance and texture coordinates final.
 *
 *          Command set 2.6: with
 *          DRAW_TEX1 a vertex is 20 floats, texture unit 1's coordinates
 *          last, generated and transformed by unit 1's registers; points
 *          sized by their distance from the eye (EXT_point_parameters).
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#version 450

/* The registers (VGPU98_OGL_R_*) */
#define R_MODELVIEW      0u
#define R_PROJECTION     16u
#define R_TEXTURE        32u
#define R_NORMAL         48u
#define R_ENABLES        60u
#define R_LIGHTS_ON      61u
#define R_CLIPS_ON       62u
#define R_MODEL_AMBIENT  65u
#define R_MODEL_FLAGS    69u
#define R_COLOR_MATERIAL 70u
#define R_MATERIAL       72u /* + 20 a face */
#define R_LIGHT          112u /* + 24 a light */
#define R_TEXGEN_MODE    304u
#define R_TEXGEN_OBJECT  305u
#define R_TEXGEN_EYE     321u
#define R_CLIP_PLANE     337u
#define R_DEPTH_RANGE    378u
#define R_VIEWPORT       389u
#define R_POINT_SIZE     393u
#define R_LINE_WIDTH     394u
#define R_UNIT1          433u /* 2.6: texture unit 1's registers (VGPU98_OGL_U_*): */
#define U_TEXTURE        0u
#define U_TEXGEN_MODE    16u
#define U_TEXGEN_OBJECT  17u
#define U_TEXGEN_EYE     33u
#define R_POINT_PARAMS   488u /* 2.6: EXT_point_parameters (VGPU98_OGL_R_POINT_*) */
#define R_POINT_MIN      489u
#define R_POINT_MAX      490u
#define R_POINT_FADE     491u
#define R_POINT_ATTEN    492u
#define R_VEC4S          124u /* the registers' uvec4s (VGPU98_OGL_R_COUNT, rounded up) */

#define EN_LIGHTING       0x00000001u
#define EN_COLOR_MATERIAL 0x00000002u
#define EN_NORMALIZE      0x00000004u
#define EN_POINT_SMOOTH   0x00010000u
#define EN_LINE_SMOOTH    0x00020000u

#define KIND_POINTS    0u
#define KIND_LINES     1u
#define KIND_TRIANGLES 2u
#define KIND_PIXELS    14u /* the device's own: an image's fragments (PIXELS), its kind << 4 */
#define KIND_CLEAR     15u /* the device's own: a quad over the scissor box, for a clear through partial masks */
#define DRAW_CLIP      0x100u
#define DRAW_STIPPLE_ON 0x200u
#define DRAW_TEX1      0x800u /* 2.6: 20-float vertices, unit 1's texture coordinates last */
#define PIXELS_BITMAP  3u

/* The registers, then each unit's texture drawn with (ogl.frag) */
layout(std140, set = 0, binding = 0) uniform Regs {
    uvec4 r4[R_VEC4S + 32u];
};
layout(std430, set = 0, binding = 1) readonly buffer Verts {
    float v[];
};
layout(std430, set = 0, binding = 2) readonly buffer Indices {
    uint ix[];
};

layout(push_constant) uniform Draw {
    uint control; /* DRAW's control, KIND_CLEAR, or KIND_PIXELS | the image's kind << 4 */
    uint height;  /* the drawable's */
    uint width;
    uint bit;     /* PIXELS of stencil: the bit drawn | the value it takes << 8 */
    vec4 pos;     /* PIXELS: window x, y, z, the fog distance; CLEAR: z in z */
    vec4 zoom;    /* PIXELS: zoom x and y, the image's width and height */
    vec4 color;   /* CLEAR, PIXELS: the colour */
    vec4 tex;     /* PIXELS: the texture coordinates */
    vec4 tex1;    /* PIXELS: unit 1's */
} d;

layout(location = 0) out vec4 o_front;
layout(location = 1) out vec4 o_back;
layout(location = 2) flat out vec4 o_front_flat;
layout(location = 3) flat out vec4 o_back_flat;
layout(location = 4) out vec4 o_tex;
layout(location = 5) out float o_fog;
layout(location = 6) out vec4 o_clip0;       /* the user's clip planes 0-3 (triangles) */
layout(location = 7) out vec2 o_clip1;       /* 4 and 5 */
layout(location = 8) flat out vec4 o_shape;  /* lines: the ends in window coordinates; points: the centre, the radius */
layout(location = 9) flat out vec4 o_shape2; /* lines: the stipple counter's start, the half width, smooth, x-major */
layout(location = 10) out vec4 o_tex1;       /* unit 1's texture coordinates */

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

vec4
R4(uint i)
{
    return vec4(RF(i), RF(i + 1u), RF(i + 2u), RF(i + 3u));
}

mat4
RM(uint i)
{
    return mat4(R4(i), R4(i + 4u), R4(i + 8u), R4(i + 12u));
}

uint
index16(uint i)
{
    const uint w = ix[i >> 1];

    return ((i & 1u) != 0u) ? (w >> 16) : (w & 0xffffu);
}

float
V(uint vi, uint k)
{
    return v[vi * (((d.control & DRAW_TEX1) != 0u) ? 20u : 16u) + k];
}

/* A vertex after the per-vertex operations */
struct Vert {
    vec4  clip;
    vec4  front;
    vec4  back;
    vec4  tex;
    vec4  tex1; /* unit 1's */
    float eyed; /* the distance from the eye (EXT_point_parameters) */
    float fog;
    float cd[6]; /* the user's clip planes' distances (at or above 0 inside) */
};

/* GL 1.1's lighting equation (section 2.13.1) for a face's material at
   register m, the normal n facing it, eye the vertex in eye coordinates. */
vec4
light_face(uint m, vec3 n, vec4 eye, vec4 color, bool track)
{
    vec4  amb  = R4(m + 0u);
    vec4  dif  = R4(m + 4u);
    vec4  spe  = R4(m + 8u);
    vec4  emi  = R4(m + 12u);
    float shin = RF(m + 16u);
    vec3  c;
    uint  on = R(R_LIGHTS_ON);
    bool  local = (R(R_MODEL_FLAGS) & 1u) != 0u;
    uint  i;

    /* Colour material: the current colour for the material's parts named. */
    if (track) {
        switch (R(R_COLOR_MATERIAL) >> 4) {
            case 0u:
                emi = color;
                break;
            case 1u:
                amb = color;
                break;
            case 2u:
                dif = color;
                break;
            case 3u:
                spe = color;
                break;
            default:
                amb = color;
                dif = color;
                break;
        }
    }
    c = emi.rgb + R4(R_MODEL_AMBIENT).rgb * amb.rgb;
    for (i = 0u; i < 8u; i++) {
        uint  l = R_LIGHT + 24u * i;
        vec4  pos;
        vec3  vp, term;
        float att = 1.0, spot = 1.0, cutoff, ndl;

        if ((on & (1u << i)) == 0u)
            continue;
        pos = R4(l + 12u);
        if (pos.w != 0.0) {
            vec3  dv = pos.xyz / pos.w - eye.xyz / ((eye.w != 0.0) ? eye.w : 1.0);
            float dist = length(dv);
            float a;

            vp  = (dist > 0.0) ? dv / dist : dv;
            a   = RF(l + 21u) + RF(l + 22u) * dist + RF(l + 23u) * dist * dist;
            att = (a > 0.0) ? 1.0 / a : 1.0;
        } else
            vp = (dot(pos.xyz, pos.xyz) > 0.0) ? normalize(pos.xyz) : pos.xyz;
        cutoff = RF(l + 20u);
        if (cutoff != 180.0) {
            vec3  dir  = vec3(RF(l + 16u), RF(l + 17u), RF(l + 18u));
            float cosv = -dot(vp, (dot(dir, dir) > 0.0) ? normalize(dir) : dir);
            float e    = RF(l + 19u);

            if (cosv >= cos(radians(cutoff)))
                spot = (e > 0.0) ? ((cosv > 0.0) ? pow(cosv, e) : 0.0) : 1.0;
            else
                spot = 0.0;
        }
        ndl  = dot(n, vp);
        term = R4(l + 0u).rgb * amb.rgb;
        if (ndl > 0.0) {
            vec3  h = local ? vp + ((dot(eye.xyz, eye.xyz) > 0.0) ? normalize(-eye.xyz) : vec3(0.0)) : vp + vec3(0.0, 0.0, 1.0);
            float ndh;

            term += ndl * R4(l + 4u).rgb * dif.rgb;
            h   = (dot(h, h) > 0.0) ? normalize(h) : h;
            ndh = dot(n, h);
            if (ndh > 0.0)
                term += ((shin > 0.0) ? pow(ndh, shin) : 1.0) * R4(l + 8u).rgb * spe.rgb;
        }
        c += att * spot * term;
    }
    return vec4(clamp(c, 0.0, 1.0), clamp(dif.a, 0.0, 1.0));
}

/* A unit's texture coordinates t: generated where on (its mode register,
   object and eye planes), then through its matrix. */
vec4
unit_tex(vec4 t, uint mat, uint gen_reg, uint obj_reg, uint eye_reg, vec4 obj, vec4 eye, vec3 n)
{
    const uint gen = R(gen_reg);
    uint       i;

    for (i = 0u; i < 4u; i++) {
        uint mode = (gen >> (4u * i)) & 15u;

        if (mode == 1u)
            t[i] = dot(R4(obj_reg + 4u * i), obj);
        else if (mode == 2u)
            t[i] = dot(R4(eye_reg + 4u * i), eye);
        else if ((mode == 3u) && (i < 2u)) {
            vec3  u = (dot(eye.xyz, eye.xyz) > 0.0) ? normalize(eye.xyz) : eye.xyz;
            vec3  rr = u - 2.0 * n * dot(n, u);
            float mm = 2.0 * sqrt(rr.x * rr.x + rr.y * rr.y + (rr.z + 1.0) * (rr.z + 1.0));

            if (mm > 0.0)
                t[i] = rr[i] / mm + 0.5;
        }
    }
    return RM(mat) * t;
}

/* Vertex vi through the per-vertex operations (or as the ICD gave it). */
Vert
vertex(uint vi)
{
    Vert       o;
    vec4       obj   = vec4(V(vi, 0u), V(vi, 1u), V(vi, 2u), V(vi, 3u));
    vec4       color = vec4(V(vi, 4u), V(vi, 5u), V(vi, 6u), V(vi, 7u));
    vec4       t     = vec4(V(vi, 12u), V(vi, 13u), V(vi, 14u), V(vi, 15u));
    const bool tex1  = (d.control & DRAW_TEX1) != 0u;
    vec4       t1    = tex1 ? vec4(V(vi, 16u), V(vi, 17u), V(vi, 18u), V(vi, 19u)) : vec4(0.0, 0.0, 0.0, 1.0);
    uint       en    = R(R_ENABLES);
    uint       i;

    if ((d.control & DRAW_CLIP) != 0u) {
        o.clip  = obj;
        o.front = clamp(color, 0.0, 1.0);
        o.back  = o.front;
        o.tex   = t;
        o.tex1  = t1;
        o.fog   = V(vi, 8u);
        o.eyed  = 0.0;
        for (i = 0u; i < 6u; i++)
            o.cd[i] = 1.0;
        return o;
    }
    {
        vec4 eye = RM(R_MODELVIEW) * obj;
        mat3 nm  = mat3(R4(R_NORMAL).xyz, R4(R_NORMAL + 4u).xyz, R4(R_NORMAL + 8u).xyz);
        vec3 n   = nm * vec3(V(vi, 8u), V(vi, 9u), V(vi, 10u));
        uint clips = R(R_CLIPS_ON);

        if (((en & EN_NORMALIZE) != 0u) && (dot(n, n) > 0.0))
            n = normalize(n);
        o.clip = RM(R_PROJECTION) * eye;
        if ((en & EN_LIGHTING) != 0u) {
            uint cm    = ((en & EN_COLOR_MATERIAL) != 0u) ? (R(R_COLOR_MATERIAL) & 3u) : 0u;

            o.front = light_face(R_MATERIAL, n, eye, color, (cm & 1u) != 0u);
            if ((R(R_MODEL_FLAGS) & 2u) != 0u)
                o.back = light_face(R_MATERIAL + 20u, -n, eye, color, (cm & 2u) != 0u);
            else
                o.back = o.front;
        } else {
            o.front = clamp(color, 0.0, 1.0);
            o.back  = o.front;
        }
        /* Each unit's texture coordinates: generated where on, then its
           matrix (unit 1's only with its coordinates in the vertex). */
        o.tex  = unit_tex(t, R_TEXTURE, R_TEXGEN_MODE, R_TEXGEN_OBJECT, R_TEXGEN_EYE, obj, eye, n);
        o.tex1 = tex1 ? unit_tex(t1, R_UNIT1 + U_TEXTURE, R_UNIT1 + U_TEXGEN_MODE, R_UNIT1 + U_TEXGEN_OBJECT,
                                 R_UNIT1 + U_TEXGEN_EYE, obj, eye, n)
                      : t1;
        o.fog  = abs(eye.z);
        o.eyed = length(eye.xyz / ((eye.w != 0.0) ? eye.w : 1.0));
        for (i = 0u; i < 6u; i++)
            o.cd[i] = ((clips & (1u << i)) != 0u) ? dot(R4(R_CLIP_PLANE + 4u * i), eye) : 1.0;
    }
    return o;
}

/* GL's viewport: x, y, width, height */
vec4
viewport()
{
    return vec4(float(int(R(R_VIEWPORT))), float(int(R(R_VIEWPORT + 1u))), float(int(R(R_VIEWPORT + 2u))),
                float(int(R(R_VIEWPORT + 3u))));
}

/* Window coordinates of clip coordinates (x, y; z's 0-1 before the depth
   range, which Vulkan's viewport applies) */
vec3
window(vec4 clip)
{
    vec4 vp  = viewport();
    vec3 ndc = clip.xyz / clip.w;

    return vec3((ndc.x + 1.0) * 0.5 * vp.z + vp.x, (ndc.y + 1.0) * 0.5 * vp.w + vp.y, (ndc.z + 1.0) * 0.5);
}

/* A point in window coordinates as Vulkan's position: for points and
   lines Vulkan's viewport is the whole window, upside down, depth 0-1 (GL
   clips them at the window alone, not at its viewport), so the depth range
   is applied here to z (0-1). */
vec4
from_window(vec2 w, float z)
{
    const float n = clamp(RF(R_DEPTH_RANGE), 0.0, 1.0);
    const float f = clamp(RF(R_DEPTH_RANGE + 1u), 0.0, 1.0);

    return vec4(2.0 * w.x / float(d.width) - 1.0, 2.0 * w.y / float(d.height) - 1.0, n + (f - n) * z, 1.0);
}

void
put(Vert a, vec4 flat_front, vec4 flat_back)
{
    o_front      = a.front;
    o_back       = a.back;
    o_front_flat = flat_front;
    o_back_flat  = flat_back;
    o_tex        = a.tex;
    o_tex1       = a.tex1;
    o_fog        = a.fog;
    o_clip0      = vec4(a.cd[0], a.cd[1], a.cd[2], a.cd[3]);
    o_clip1      = vec2(a.cd[4], a.cd[5]);
}

/* Nothing drawn: every corner at one place. */
void
none()
{
    gl_Position  = vec4(0.0, 0.0, 0.0, 1.0);
    o_front      = vec4(0.0);
    o_back       = vec4(0.0);
    o_front_flat = vec4(0.0);
    o_back_flat  = vec4(0.0);
    o_tex        = vec4(0.0);
    o_tex1       = vec4(0.0);
    o_fog        = 0.0;
    o_clip0      = vec4(1.0);
    o_clip1      = vec2(1.0);
    o_shape      = vec4(0.0);
    o_shape2     = vec4(0.0);
}

/* Inside the view volume and the user's planes (a point's clip). */
bool
inside(Vert a)
{
    uint i;

    if ((a.clip.x < -a.clip.w) || (a.clip.x > a.clip.w) || (a.clip.y < -a.clip.w) || (a.clip.y > a.clip.w) ||
        (a.clip.z < -a.clip.w) || (a.clip.z > a.clip.w))
        return false;
    for (i = 0u; i < 6u; i++) {
        if (a.cd[i] < 0.0)
            return false;
    }
    return true;
}

Vert
mix_vert(Vert a, Vert b, float t)
{
    Vert o;
    uint i;

    o.clip  = mix(a.clip, b.clip, t);
    o.front = mix(a.front, b.front, t);
    o.back  = mix(a.back, b.back, t);
    o.tex   = mix(a.tex, b.tex, t);
    o.tex1  = mix(a.tex1, b.tex1, t);
    o.eyed  = mix(a.eyed, b.eyed, t);
    o.fog   = mix(a.fog, b.fog, t);
    for (i = 0u; i < 6u; i++)
        o.cd[i] = mix(a.cd[i], b.cd[i], t);
    return o;
}

/* The line a-b clipped against the view volume and the user's planes:
   the part kept, t0 to t1; false when none is. */
bool
clip_line(Vert a, Vert b, out float t0, out float t1)
{
    uint p;

    t0 = 0.0;
    t1 = 1.0;
    for (p = 0u; p < 12u; p++) {
        float da, db;

        if (p < 6u) {
            float sa = ((p & 1u) != 0u) ? -1.0 : 1.0;

            da = a.clip.w + sa * a.clip[p >> 1];
            db = b.clip.w + sa * b.clip[p >> 1];
        } else {
            da = a.cd[p - 6u];
            db = b.cd[p - 6u];
        }
        if ((da < 0.0) && (db < 0.0))
            return false;
        if (da < 0.0)
            t0 = max(t0, da / (da - db));
        else if (db < 0.0)
            t1 = min(t1, da / (da - db));
    }
    return t0 < t1;
}

void
main()
{
    const uint kind = d.control & 15u;
    const uint id   = uint(gl_VertexIndex);
    const uint corner = id % 6u;
    /* A quad's corners: two triangles, (0, 1, 2) and (2, 1, 3) */
    const uint cn    = (corner < 3u) ? corner : ((corner == 3u) ? 2u : ((corner == 4u) ? 1u : 3u));
    const vec2 side  = vec2(((cn & 1u) != 0u) ? 1.0 : -1.0, ((cn & 2u) != 0u) ? 1.0 : -1.0);

    if (kind == KIND_CLEAR) {
        none();
        gl_Position = vec4(side, d.pos.z, 1.0);
        return;
    }
    if (kind == KIND_PIXELS) {
        /* The image's rectangle in window coordinates (Vulkan's viewport
           the whole drawable, upside down): zoomed from the raster
           position; a bitmap's at its floor, unzoomed (GL 1.1, 3.6.4 and
           3.7). ogl.frag finds each fragment's pixel. */
        vec2 a, b;

        if (((d.control >> 4) & 15u) == PIXELS_BITMAP) {
            a = floor(d.pos.xy);
            b = a + d.zoom.zw;
        } else {
            a = d.pos.xy;
            b = d.pos.xy + d.zoom.xy * d.zoom.zw;
        }
        none();
        {
            /* A pixel's margin around it: where its edge falls on pixel
               centres, ogl.frag's test decides by GL's rule, not Vulkan's
               edge rule. */
            const vec2 lo = min(a, b) - 1.0, hi = max(a, b) + 1.0;
            const vec2 w  = vec2((side.x > 0.0) ? hi.x : lo.x, (side.y > 0.0) ? hi.y : lo.y);

            gl_Position = vec4(2.0 * w.x / float(d.width) - 1.0, 2.0 * w.y / float(d.height) - 1.0, d.pos.z, 1.0);
        }
        return;
    }
    if (kind == KIND_TRIANGLES) {
        Vert a = vertex(index16(id));

        put(a, a.front, a.back);
        o_shape     = vec4(0.0);
        o_shape2    = vec4(0.0);
        gl_Position = vec4(a.clip.xy, 0.5 * (a.clip.z + a.clip.w), a.clip.w);
        return;
    }
    if (kind == KIND_POINTS) {
        const uint vi = index16(id / 6u);
        Vert  a = vertex(vi);
        bool  smooth_on = (R(R_ENABLES) & EN_POINT_SMOOTH) != 0u;
        float size = RF(R_POINT_SIZE);

        /* 2.6, EXT_point_parameters: the ICD's own points carry their size
           (and their fade in alpha); the device's are sized by their
           distance from the eye, clamped, faded below the threshold */
        if ((d.control & DRAW_CLIP) != 0u) {
            if (V(vi, 11u) > 0.0)
                size = V(vi, 11u);
        } else if ((R(R_POINT_PARAMS) & 1u) != 0u) {
            const float den = RF(R_POINT_ATTEN) + RF(R_POINT_ATTEN + 1u) * a.eyed + RF(R_POINT_ATTEN + 2u) * a.eyed * a.eyed;

            if (den > 0.0)
                size *= sqrt(1.0 / den);
            size = min(max(size, RF(R_POINT_MIN)), RF(R_POINT_MAX));
            if (size < RF(R_POINT_FADE)) {
                const float f = size / RF(R_POINT_FADE);

                a.front.a *= f * f;
                a.back.a *= f * f;
                size = RF(R_POINT_FADE);
            }
        }
        size = clamp(size, 1.0, 64.0);
        vec3  w;
        vec2  c;
        float half_size;

        if (!inside(a)) {
            none();
            return;
        }
        w = window(a.clip);
        if (smooth_on) {
            /* Its coverage in ogl.frag: a disc of the size, around the
               exact place, a pixel's margin around it. */
            c         = w.xy;
            half_size = 0.5 * size + 1.0;
            o_shape   = vec4(c, 0.5 * size, 0.0);
            o_shape2  = vec4(0.0, 0.0, 1.0, 0.0);
        } else {
            /* GL 1.1, 3.3: the size rounded; an odd size centred on a
               pixel's centre, an even one on a pixel's corner. */
            size      = max(floor(size + 0.5), 1.0);
            c         = (mod(size, 2.0) != 0.0) ? floor(w.xy) + 0.5 : floor(w.xy + 0.5);
            half_size = 0.5 * size;
            o_shape   = vec4(c, 0.5 * size, 0.0);
            o_shape2  = vec4(0.0);
        }
        put(a, a.front, a.back);
        gl_Position = from_window(c + side * half_size, w.z);
        return;
    }
    /* Lines: the provoking vertex is the second. */
    {
        const uint prim = id / 6u;
        Vert       a    = vertex(index16(2u * prim));
        Vert       b    = vertex(index16(2u * prim + 1u));
        bool       smooth_on = (R(R_ENABLES) & EN_LINE_SMOOTH) != 0u;
        float      width = clamp(RF(R_LINE_WIDTH), 1.0, 16.0);
        float      start = ((d.control & DRAW_STIPPLE_ON) != 0u) ? V(index16(2u * prim), 11u) : 0.0;
        float      t0, t1;
        vec3       wa, wb;
        vec2       dir, off, along = vec2(0.0);
        bool       xmajor;
        Vert       e;

        if ((d.control & DRAW_CLIP) == 0u) {
            if (!clip_line(a, b, t0, t1)) {
                none();
                return;
            }
            if ((t0 > 0.0) || (t1 < 1.0)) {
                Vert na = mix_vert(a, b, t0);
                Vert nb = mix_vert(a, b, t1);

                a = na;
                b = nb;
            }
        }
        wa  = window(a.clip);
        wb  = window(b.clip);
        dir = wb.xy - wa.xy;
        if (dot(dir, dir) < 1e-12) {
            none();
            return;
        }
        xmajor = abs(dir.x) >= abs(dir.y);
        if (smooth_on) {
            /* A rectangle of the width along the line; its coverage in
               ogl.frag, a pixel's margin around it. */
            vec2 u = normalize(dir);

            off   = vec2(-u.y, u.x) * (0.5 * width + 1.0);
            along = u;
        } else {
            /* GL 1.1, 3.4.2: the width rounded; a column of fragments for
               an x-major line, a row for a y-major one. */
            width = max(floor(width + 0.5), 1.0);
            off   = xmajor ? vec2(0.0, 0.5 * width) : vec2(0.5 * width, 0.0);
        }
        e = ((cn & 1u) != 0u) ? b : a;
        put(e, b.front, b.back);
        o_shape     = vec4(wa.xy, wb.xy);
        o_shape2    = vec4(start, 0.5 * width, smooth_on ? 1.0 : 0.0, xmajor ? 1.0 : 0.0);
        gl_Position = from_window((((cn & 1u) != 0u) ? wb.xy + along : wa.xy - along) + side.y * off,
                                  ((cn & 1u) != 0u) ? wb.z : wa.z);
    }
}
