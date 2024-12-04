
# Transformer



https://edstem.org/us/courses/62772/discussion/5759163

You are on the right track, but remember from the README:

https://edstem.org/us/courses/62772/discussion/5724833

This architecture most closely resembles the Perceiver model, where in our setting, the "latent array" corresponds to the target waypoint query embeddings (nn.Embedding), while the "byte array" refers to the encoded input lane boundaries.
In the diagram shown in the Perceiver, we are "projecting a high-dimensional input byte array to a fixed-dimensional latent bottleneck". As a hint, I'd revisit the parameter descriptions within theTransformerDecoder.

# CNN


Hi, you can try to reduce:

the kernel size:

kernel 3x3 from 128 channels to 256 channels: 3x3x128x256+256 = 295,168 parameters.

kernel 7x7 from 128 channels to 256 channels: 7x7x128x256+256 = 1,605,888 parameters.

the number of channels:

kernel 3x3 from 128 channels to 256 channels: 3x3x128x256+256 = 295,168 parameters.

kernel 3x3 from 256 channels to 512 channels: 3x3x256x512+512 = 1,180,160 parameters.

For the 20MB model size, you can have approx. 5,242,880 parameters.



I think a kernel size of 3 and the largest channel size of 256 can work well for this assignment.


## Start Task 3, getting lower 