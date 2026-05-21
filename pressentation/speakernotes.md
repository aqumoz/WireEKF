# Speaker Notes: Extended Kalman Filter Slides

## Slide 1: State Estimation — The Need for Filtering

**Key Points:**
- Start by framing the problem: we have a physics model that isn't perfect, and we have sensor measurements that are noisy
- Explain the three sources of model error:
  - **Unmodeled forces**: Our XPBD model doesn't capture everything. There might be air resistance, friction at contact points, or other effects we haven't included.
  - **Uncertain material parameters**: We don't know the exact stiffness and damping of the wire. We can estimate them, but they're never perfect.
  - **Numerical discretization errors**: Our simulation uses finite timesteps and iterative constraint solving. This introduces small errors that accumulate.

- The vision system (TrackDLO) gives us measurements, but they're also noisy. Vision-based tracking can have occlusions, calibration errors, etc.

- The core insight: **neither source is perfect**, but together they can complement each other.
  - If the model is wrong but we trust it completely, our estimate drifts from reality.
  - If we trust measurements completely, we're fighting noise and lose the benefit of our physics model.
  - The solution is to **fuse** them intelligently.

**Transition:** "This is exactly what the Extended Kalman Filter does..."

## Slide 2: Extended Kalman Filter — Two-Step Algorithm

**Key Points:**

**State Vector:**
- Our state includes position, velocity, and acceleration for each of the n nodes
- This is a relatively small state space compared to FEM, making it computationally tractable
- By tracking velocity and acceleration, the filter can smooth out measurement noise

**Prediction Step:**
- We call our XPBD physics model to predict where the wire will be in the next timestep
- The prediction function `f()` is nonlinear (that's why it's "extended" Kalman — it handles nonlinearity)
- We also propagate the uncertainty (covariance P). Even if we're uncertain now, we become *more* uncertain after predicting forward
- Q is the **process noise covariance** — it represents our belief in how much the model prediction might be wrong
  - Small Q: "I trust my model"
  - Large Q: "My model is unreliable, so I expect larger errors"

**Update Step:**
- Walk through the 5 equations one by one:

  1. **Innovation (y_k)**: This is the **gap** between what we measured and what we predicted. If they match, innovation is zero.
  
  2. **Innovation Covariance (S_k)**: Combines our prediction uncertainty (P) and measurement noise (R). This tells us how much total uncertainty exists.
  
  3. **Kalman Gain (K_k)**: This is the **magic**: it's a weighting factor that tells us how much to trust the measurement vs. the prediction.
     - If R is small (measurements are good), K_k is larger → we shift more toward the measurement
     - If R is large (measurements are bad), K_k is smaller → we stick closer to the prediction
  
  4. **Updated State**: We correct our prediction by adding the Kalman gain times the innovation. We're pulling our estimate toward the measurement, but only by an amount that makes sense given the uncertainties.
  
  5. **Updated Covariance**: After getting a measurement, we become *more* certain. The covariance shrinks.

**Key Insight:** The ratio Q/R controls the balance. If you want to trust measurements more, decrease R or increase Q.

---

## Slide 3: WireEKF.py — Implementation Details

**Key Points:**

**Core Components:**

- **State Vector**: Position, velocity, acceleration per node. This gives us 9 states per node (3D for each). For 10 nodes, that's 90-dimensional state space — large but manageable.

- **Constant-Acceleration Model**: The kinematic prediction is simple:
  - Position changes based on current velocity and acceleration
  - Velocity changes based on current acceleration
  - This is the simplest assumption (velocity isn't changing). The XPBD model will apply actual forces to correct this.

- **XPBD Callback**: The real physics lives here. During prediction, we don't just do kinematics — we call the full XPBD solver to get physically realistic predictions. This makes the filter much better than a naive kinematic model.

- **Position-Only Measurement**: We only measure positions from the vision system, not velocities or accelerations. The filter infers those from position changes over time. This is common and works well.

**Tuning Parameters:**

- **Q (Process Noise)**: 
  - If high: "The model makes big mistakes, so I don't trust predictions much"
  - If low: "The model is pretty good, so I trust predictions"
  - Example: We might set Q high if we know there are external disturbances we haven't modeled

- **R (Measurement Noise)**:
  - If high: "Vision measurements are unreliable"
  - If low: "Vision measurements are very accurate"
  - You'd lower R if you've invested in a high-quality camera system

**Trade-off:** The Q/R ratio is what really matters. A Q of 0.9 and R of 0.01 means we trust measurements 90x more than the model, which would make the estimate follow the measurements closely. This is good if the model has large errors but bad if the measurements are noisy.

---

## Slide 4: Why Extended Kalman Filter for Wire Tracking?

**Key Points:**

**Challenges in DLO State Estimation:**

- **Model approximation**: We simplified the wire to XPBD, which is good but not perfect. Real wires have properties we don't model.

- **Noisy sparse measurements**: Vision systems can fail. Maybe a node is occluded. Maybe the lighting is bad. We can't rely on one measurement.

- **Nonlinear dynamics**: Wire bending and gravity make this inherently nonlinear. A linear Kalman filter won't work; we need the extended version (which linearizes around the current estimate).

- **Real-time constraints**: We need state estimates *now* for robot control. We can't run expensive FEM simulations. EKF is computationally cheap.

**Why EKF Solves This:**

- **Handles nonlinearity**: By using the Jacobian matrix F and H, we linearize around the current estimate. Not perfect, but much better than assuming linearity.

- **Fuses sensor data optimally**: The mathematics of the Kalman gain is optimal (in a least-squares sense) for linear systems. For nonlinear systems, EKF is the best practical option.

- **Uncertainty tracking**: We get P, the covariance matrix. This tells us how confident we are in each state. High P in position means we're uncertain; low P means confident.

- **Computationally efficient**: O(n³) per timestep. Scales well to 10-50 nodes. FEM would be prohibitively expensive.

- **Adaptive tuning**: By adjusting Q and R, we can adapt the filter's behavior to different scenarios (different materials, different measurement qualities, different disturbances).

**Bottom Line:** EKF gives us the best state estimate we can get *in real time* given noisy model predictions and noisy measurements. This is essential when a robot arm is moving and we need accurate feedback.

---

## Slide 5: EKF Validation — Animation Output

**Key Points:**

**Current Stage:**
- We're validating the whole pipeline with simulation data from Simulink
- The wire is suspended from one end (the robot end-effector) and hanging down under gravity
- Our synthetic "measurements" come from the same XPBD model, but we add noise to make it realistic
- We're tuning Q and R to see how the filter responds
  - Show what happens with high Q (filter follows measurements closely)
  - Show what happens with low Q (filter stays close to model)

**The Animation:**
- The image/video shows **three colored traces**:
  - **Green (Model)**: What the XPBD model predicts alone, without any measurement feedback. You'll see it can drift over time.
  - **Red (Measurements)**: The noisy sensor data we receive from vision. Jumpy, noisy.
  - **Blue (EKF Estimate)**: The filtered result. Should be smoother than red, more accurate than green.
- Show how the blue trace is a "middle ground" — it follows the red measurements but is smoothed by the green model predictions.

**Next Phase — Real Hardware:**
- Once we're confident in simulation, we swap out the Simulink data for real TrackDLO output
- TrackDLO runs in real-time and sends us vision-based node positions
- Same EKF code, same Q/R tuning (or close), but now with real robot and real wire
- The robot can use these state estimates to decide its next move

**Closing:** "The beauty of this approach is that the EKF acts as a bridge between our imperfect physics model and our noisy sensors, giving us accurate, real-time state estimates that the robot can trust."
