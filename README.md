## JAX Ray Tracing

This is the beginnings of a simple ray tracing implementation using JAX for fun and learning. It's also a opportunity to do misguided things on TPU for novelty value, like (eventually) use a TPU for massively parallel path tracing, or use Pallas like a shader language.

Personally, I'm hoping that applying tools to the wrong problems will reveal alternative perspectives vs learning about things the "right" way. I'm not stopping to think if I should, but wondering about whether I could (can). :)

### Why Ray Tracing?

A university course I did many moons ago sparked an interest in ray tracing for me. Some of my more clever friends (than I) went on to work in the industry, but I think about ray tracing every now and then. This seems like a fun way to do that.

I'm familiar with the math (thanks uni degree), but haven't "worked it out on paper" in decades. Luckily, Gemini (and Claude) have proved amazing at polishing my attempts to vectorize Möller–Trumbore and other algos.

### TODO list

The desired end-state is to have the following work:

- [X] Basic first-hit ray tracing.
- [ ] Tracing reflections and refractions (transparency).
- [ ] Sphere and hyperplanes.
- [X] Triangles and triangle meshes.
- [X] Diffusion lighting.
- [ ] Phong shading (ambient + diffusion + specular)
- [X] Hard shadows.
- [ ] Soft shadows.
- [X] Point lights.
- [ ] Area lights.
- [ ] Naive path tracing.
- [X] Basic distributed tracing using JAX SPMD (currently tested to work on GPU)
- [ ] Adaptive super-sampling (maybe as a Pallas kernel?)
- [ ] Separately optimizing for TPU and GPU execution.

Some of these TODOs may be difficult / impossible to get working (well) on TPU, given that it's not exactly designed for graphics algorithms (something something not stopping to think if I should).

On GPU/TPU optimization, Gemini and Claude has applied a critical eye to the code and identified a *lot* of deficiencies. In that sense, this could be a good base to try out different optimizations that work better/worse on GPU vs TPU.

### Other related work

A cursory Internet search reveals at least a couple of projects that use JAX for ray tracing:

- [jax-raytracing](https://github.com/schmidtdominik/jax-raytracing)
- [Ray Tracing in JAX](https://kayleegeorge.github.io/blog/jax-ray-tracer/)

I've not seen another project attempt rendering full triangle meshes using JAX (Stanford Bunny ftw). I'd be pleased if this was the first, as unlikely as that would be!