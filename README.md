# Proto_Design_GA
Application of Response Surface Method and Genetic Algorithm in the Design of High-Efficiency Prototype Vehicle

The overall design of the vehicle is divided into two segments: the side and the top profiles. These profiles are then parameterized using a three degree Bezier curve. The position of the control points of these curves are optimized using the Genetic Algorithm for minimizing the drag force. The algorithm utilizes the Kriging surrogate model for its operation. The surrogate model is trained on RANS simulations with the k-omega SST turbulence model.
