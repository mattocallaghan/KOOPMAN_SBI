# Distilled Simulation-Based Inference with Koopman-Lifted Dynamics

Work in Progress

This project implements a method for one-shot inference in complex simulation-based inference (SBI) tasks. It uses Koopman-lifted dynamics to estimate conditional probabilities efficiently, enabling faster inference while maintaining accuracy in complex simulations.

---

## Overview

Simulation-based inference is useful when likelihoods are intractable or expensive to compute. Traditional SBI methods often require many simulations, which can be slow for complex models.

This project explores a distillation approach:

- Train a student model to mimic a more expensive SBI teacher.
- Use Koopman-lifted dynamics to represent nonlinear system dynamics linearly in a higher-dimensional space.
- Enable one-shot inference for conditional probabilities.
- Reduce simulation costs and speed up inference.

---

## Features

- Distillation for simulation-based inference
- Koopman-lifted dynamics for linear representation of nonlinear systems
- One-shot inference for fast conditional probability estimation
- Flexible integration with existing SBI pipelines

---

## Installation

Work in progress – installation instructions may change.

```bash
git clone https://github.com/yourusername/koopman-sbi-distillation.git
cd koopman-sbi-distillation
pip install -r requirements.txt
