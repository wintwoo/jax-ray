## JAX Ray Tracing

This is the beginnings of a simple ray tracing implementation using JAX for fun and learning. I'm also interested in applying JAX's hardware acceleration and distributed processing for ray tracing. GPUs would be the natural fit here - but I also thought it would be an interesting to see how much I could accelerate using other XPUs (e.g. TPUs). This is an exploration, so practicality is not the goal here! 

### Why Ray Tracing?

A university course I did many moons ago sparked an interest in ray tracing for me. Some of my cleverer friends (than I) went on to work in the industry, but I think about ray tracing every now and then. 

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

Some of these TODOs may be difficult / impossible to get working (well) on XPU. For example, certain data structures like BSP trees seem quite far-fetched to implement.

On GPU/TPU optimization, Gemini and Claude has applied a critical eye to the code and identified a *lot* of deficiencies. In that sense, this could be a good base to try out different optimizations that work better/worse on GPU vs TPU.

### Other related work

A cursory Internet search reveals at least a couple of projects that use JAX for ray tracing:

- [jax-raytracing](https://github.com/schmidtdominik/jax-raytracing)
- [Ray Tracing in JAX](https://kayleegeorge.github.io/blog/jax-ray-tracer/)

I've not seen another project attempt rendering full triangle meshes using JAX (Stanford Bunny ftw). I'd be pleased if this was the first, as unlikely as that would be!
