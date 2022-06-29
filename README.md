Let’s learn Intel oneAPI AI Analytics Toolkit
How to Achieve End-to-End Performance for AI Workloads



Note: The purpose of this blog is to make the learning curve of the Intel OneAPI AI Toolkit easier and help you guys understand how each toolkit works and how to implement them. Also at the end of each topic, you may find the official Intel getting started page/example notebooks which are a great resource for beginners. Happy learning!

## INDEX
What’s OneAPI

AI Toolkit

Requirements and Intel DevCloud

Intel Distribution for Python

Intel Distribution for Modin

Intel Extension for Scikit-Learn

XGBoost Optimized by Intel

Intel Optimization for TensorFlow

Intel Optimization for PyTorch

Model Zoo for Intel Architecture

Intel Neural Compressor


Want to reduce your model training and inference time but can’t afford expensive GPUs? Looking for ways to accelerate your end-to-end data science and analytics pipelines on various intel architectures? Fret not, Intel’s oneAPI AI Analytics Toolkit (AI Kit) provides intel XPU’s based optimizations for popular tools such as Python, Pandas (Modin), scikit-learn, Tensorflow, PyTorch. This toolkit is based on the OneAPI programming model. Wait, what’s OneAPI?

## What’s OneAPI?
OneAPI is a cross-architecture language based on C++ and SYCL standards. It provides powerful libraries designed for an acceleration of domain-specified functions. The bold vision of OneAPI is to have a cross-architecture, cross vendors software portability while providing you with all the performance your need. So no matter what devices and accelerators your system might have or what languages and libraries each of these devices are using, middleware or frameworks applications and workloads, OneAPI is bridging, abstracting and bringing all these devices to a common ground.

The industry-wide spec of OneAPI defines a low-level abstraction layer inside the software stack and you can use the set of optimized libraries for various domains. for example, oneDNN the deep neural network library or oneMKL, the Math Kernal Library and more. In addition, for cross-architecture direct programming the Data Parallel C++ or DPC++ language is included. It is built on open standards and specifications. The Intel implementation of OneAPI includes many additional libraries, compilers, and analyzers. It is arranged in toolkits for specific application domains and can be downloaded in many ways or used remotely on the Intel DevCloud.

Think of it as a programming model which brings many different languages, libraries, and hardware to a common ground where memory spaces could be shared, code could be ported, re-used and tools could work across architectures.

Intel® oneAPI AI Analytics Toolkit (AI Kit)
Intel AI Analytics Toolkit contains tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. It is built using OneAPI libraries for compute optimizations.

Using this toolkit, you can:

Deliver high-performance, deep-learning training on Intel® XPUs and integrate fast inference into your AI development workflow with Intel®-optimized, deep-learning frameworks for TensorFlow* and PyTorch*, pre-trained models, and low-precision tools.
Achieve drop-in acceleration for data preprocessing and machine-learning workflows with compute-intensive Python* packages, Modin*, scikit-learn*, and XGBoost, optimized for Intel.
Gain direct access to analytics and AI optimizations from Intel to ensure that your software works together seamlessly.
Requirements and Intel DevCloud
Unfortunately not every tool in Intel AI Toolkit supports Windows operating system and you will have to make sure that you satisfy other system requirements as well. For example, Intel AI Toolkit requires CPU: Intel Core Gen10 Processor or Intel Xeon Scalable Performance processors and Support for Intel Graphics (GPU) are yet to be added in a future release.



As our aim is to get started with the AI Toolkit with ease, I recommend using Intel DevCloud for OneAPI. Intel DevCloud is a development sandbox to learn about programming cross-architecture applications such as AI Analytics Toolkit. It’s completely free to use 120 days and we get around 200 GB of file storage, 192 GB RAM with various Intel Xeon processors and FPGAs. What’s more? They provide specific kernels for each toolkits making our learning experience seamless. To use the DevCloud you will first have to create your Intel DevCloud account and verify your account. Once verified, go to https://devcloud.intel.com/oneapi/get_started/ and scroll down. You will see something like this:


Click Launch JupyterLab* and you will be redirected to the Jupyter Environment where you can create and execute your Python Jupyter Notebook.

Here are the Toolkits OS support details. Just in case, if you want to try the AI Toolkit locally on your PC.


## Intel® Distribution for Python*
Intel Distribution for Python provides accelerated performance for numerical computing and data science on Intel architectures. With the distribution, math and statistical packages such as NumPy, SciPy, and scikit-learn are linked with Intel’s performance libraries for near-native code speeds. Libraries include:

Math Kernel Library (Intel MKL) with optimized BLAS, LAPACK, FFT, and random number generators
Message Passing Interface (Intel MPI)
Thread Building Blocks (Intel TBB)
Data Analytics Acceleration Library (Intel DAAL)
Numerical and Scientific: NumPy, SciPy, Numba, numexpr
Data Analytics: pyDAAL, daal4py
Parallelism: smp, tbb, mpi4
Installation
1. Update conda

conda update conda
2. Create new empty environment, Python 3, and Intel’s Python 3.

conda create -n intel -c intel intelpython3_full python=3
3. Enter environment

activate intel
4. Install packages

conda install numpy scipy scikit-learn pydaal tbb4py
In general, you do not need to change your Python code to take advantage of the improved performance Intel’s Python Distribution provides. However, for random number generators, we recommend using the MKL-based random number generator numpy.random_intel as a drop-in replacement for numpy.random.

The update 1 of the Intel® Distribution for Python* 2017 Beta introduces numpy.random_intel, an extension to numpy which closely mirrors the design of numpy.random and uses Intel® MKL's vector statistics library to achieve a significant performance boost.

Unlocking the performance benefits is as simple as replacing numpy.random with numpy.random_intel:


For machine learning, Intel Distribution for Python provides deep learning software such as Caffe and Theano, as well as classic machine learning libraries, such as scikit-learn and pyDAAL (which implements Python bindings to Intel DAAL).


## Intel® Distribution of Modin
Modin is a parallel and distributed data frame system that enables scalable Data Analytics. This library is OmniSci* in the back end and provides accelerated analytics on Intel® platforms. It’s fully compatible with the pandas API. Using Dask and Ray, Intel Distribution of Modin transparently distributes the data and computation across available cores, unlike pandas, which only uses one core at a time.

All you need to do to accelerate your pandas is to change a single line of code: import modin.pandas as pd instead of import pandas as pd.

What’s amazing about Modin is you can easily scale your workload to the cloud or a high-performance computing environment as needed.


The with statement creates a remote execution context in the cloud, AWS in this case, with credentials provided by the user in aws_credentials.json. Modin automatically connects to AWS, spawns a cluster for distributed computation, provisions the Modin environment, then remotely executes all the Modin statements within the with clause.


## Intel® Extension for Scikit-learn*
Intel® Extension for Scikit-learn* uses patching to accelerate Scikit-learn and still have full conformance with all Scikit-Learn APIs and algorithms.


We have only added two extra lines of code and the rest is the same as what we do for Scikit-learn.

scikit-learn-intelex/examples/notebooks at master · intel/scikit-learn-intelex
This folder contains examples of python notebooks that use Intel(R) extension for Scikit-learn for popular datasets.
github.com

## XGBoost Optimized by Intel
Intel® AI Analytics Toolkit includes XGBoost with Intel optimizations for XPU. There are multiple ways to get the toolkit and its components. It is distributed through several channels — Anaconda, Docker containers, Package managers (Yum, Apt, Zypper) and an online/offline installer from Intel. To download XGBoost from the Intel® oneAPI AI Analytics Toolkit, visit here and choose the installation method of your choice. You can find more detailed information about the toolkit here.

All we need to do is to install Intel® oneAPI AI Analytics Toolkit and run the XGBoost program. We don’t have to change anything inside the script.

import xgboost as xgb
If you get this message:

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex) 
when importing Xgboost, it means that you’re using the intel optimized version for XgBoost!


## Intel® Optimization for TensorFlow*
TensorFlow* is a leading deep learning and machine learning framework, which makes it important for Intel and Google to ensure that it is able to extract maximum performance from Intel’s hardware offering. Intel developed a number of optimized deep learning primitives that can be used inside the different deep learning frameworks to ensure that we implement common building blocks efficiently. In addition to matrix multiplication and convolution, these building blocks include:

Direct batched convolution
Inner product
Pooling: maximum, minimum, average
Normalization: local response normalization across channels (LRN), batch normalization
Activation: rectified linear unit (ReLU)
Data manipulation: multi-dimensional transposition (conversion), split, concat, sum and scale.
To get hands-on practice, follow this github repository with instructions on how to run the notebook.

models/docs/notebooks/transfer_learning/tf_image_classification at master · IntelAI/models
This notebook uses transfer learning with multiple TF Hub image classifiers, TF datasets, and custom image datasets…
github.com

## Intel® Optimization for PyTorch*
Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Intel® Extension for PyTorch* is loaded as a Python module for Python programs or linked as a C++ library for C++ programs. Users can enable it dynamically in script by importing intel_extension_for_pytorch. It covers optimizations for both imperative mode and graph mode. Optimized operators and kernels are registered through PyTorch dispatching mechanism. These operators and kernels are accelerated from native vectorization feature and matrix calculation feature of Intel hardware. During execution, Intel® Extension for PyTorch* intercepts invocation of ATen operators, and replace the original ones with these optimized ones. In graph mode, further operator fusions are applied manually by Intel engineers or through a tool named oneDNN Graph to reduce operator/kernel invocation overheads, and thus increase performance.

Installation
You can use either of the following 2 commands to install Intel® Extension for PyTorch*.

python -m pip install intel_extension_for_pytorch
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
Note: Intel® Extension for PyTorch* has PyTorch version requirement. Please check more detailed information via the URL below.

More installation methods can be found at Installation Guide.

The following code snippet shows an inference code with FP32 data type. More examples, including training and C++ examples, are available at Example page.


GitHub - intel/intel-extension-for-pytorch: A Python package for extending the official PyTorch…
Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Most of…
github.com

## Model Zoo for Intel® Architecture
Open Model Zoo for OpenVINO™ toolkit delivers a wide variety of free, pre-trained deep learning models and demo applications that provide full application templates to help you implement deep learning in Python, C++, or OpenCV Graph API (G-API). Models and demos are available in the Open Model Zoo GitHub repo and licensed under Apache License Version 2.0.



## Intel® Neural Compressor
Deep neural networks (DNNs) show state-of-the-art accuracy in a wide range of computation tasks. However, they still face challenges during application deployment due to their high computational complexity of inference. Low precision is one of the key techniques that help conquer the problem.

Intel® Neural Compressor is an open-source Python* library designed to help you quickly deploy low-precision inference solutions on popular deep-learning frameworks such as TensorFlow*, PyTorch*, MXNet*, and ONNX* (Open Neural Network Exchange) runtime. The tool automatically optimizes low-precision recipes for deep-learning models to achieve optimal product objectives, such as inference performance and memory usage, with expected accuracy criteria.

Install:

install stable version from pip
pip install neural-compressor
Quantization with Python API


Congratulations, you just learned about Intel® AI Analytics Toolkit and its tools which has an amazing potential to accelerate your future Data Science projects!

P.S: This was my first ever blogging experience and I’m so glad that I did it. I’ve tried my level best to not include any mistakes in this blog, but if you do find any mistakes, feel free to comment below or contact me on any one of my social media platforms. I would really appreciate the effort!😄

