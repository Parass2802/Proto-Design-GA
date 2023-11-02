# Proto_Design_GA
Application of Response Surface Method and Genetic Algorithm in the Design of High-Efficiency Prototype Vehicle

The overall design of the vehicle is divided into two segments: the side and the top profiles. These profiles are then parameterized using a three degree Bezier curve. The position of the control points of these curves are optimized using the Genetic Algorithm for minimizing the drag force. The algorithm utilizes the Kriging surrogate model for its operation. The surrogate model is trained on RANS simulations with the k-omega SST turbulence model.

![pressure-side](https://github.com/Parass2802/Proto-Design-GA/assets/149015075/876dd834-17aa-4ce9-8025-884a54aaa240)
![velocity-side](https://github.com/Parass2802/Proto-Design-GA/assets/149015075/0b6a4ddd-1292-4924-97f3-7366156be70c)
