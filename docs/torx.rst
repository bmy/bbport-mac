ExecuTorch backend
==================

Torx is Mesa's ExecuTorch backend: it can make use of NPUs to accelerate ML
inference, implemented as a custom backend for the
`ExecuTorch <https://github.com/pytorch/executorch>`_ on-device AI framework.

Build ExecuTorch
----------------

Torx requires the ExecuTorch runtime library (``libexecutorch.so``) at build
time, and the ExecuTorch Python wheel at run time (for exporting models).

Clone and initialise the repository:

.. code-block:: console

   ~ $ git clone https://github.com/pytorch/executorch.git
   ~ $ cd executorch
   executorch $ git submodule update --init --recursive

Install the Python wheel and the runtime library together using a single build
via ``pip install``. A CPU-only PyTorch must already be installed
(``pip install torch --index-url https://download.pytorch.org/whl/cpu``).

.. code-block:: console

   executorch $ pip install -r requirements-dev.txt
   executorch $ CMAKE_ARGS="-DEXECUTORCH_BUILD_SHARED=ON \
       -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
       -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
       -DEXECUTORCH_BUILD_COREML=OFF" \
       pip install --break-system-packages --no-build-isolation .

   # Install the runtime library and headers system-wide
   executorch $ ET_BUILD=$(ls -d pip-out/temp.*/cmake-out | head -1)
   executorch $ cmake --build $ET_BUILD -j$(nproc)
   executorch $ sudo cmake --install $ET_BUILD

The ``--no-build-isolation`` flag is needed so the build can find the
system-installed PyTorch. The ``EXECUTORCH_BUILD_*=OFF`` flags
disable features which aren't needed but are known to break the build
with recent compiler versions.

Verify the installation provides a working pkg-config file:

.. code-block:: console

   ~ $ pkg-config --cflags --libs executorch

To install to a non-standard prefix instead, pass
``-DCMAKE_INSTALL_PREFIX=/path/to/install`` in ``CMAKE_ARGS`` and set
``PKG_CONFIG_PATH`` accordingly:

.. code-block:: console

   ~ $ export PKG_CONFIG_PATH=/path/to/install/lib64/pkgconfig

Build Mesa
----------

Build Mesa as usual, with the ``-Dtorx=true`` argument. Make sure at least one Gallium driver with NPU support is enabled.

.. code-block:: console

   ~ $ cd mesa
   mesa $ meson setup build -Dgallium-drivers=ethosu -Dvulkan-drivers= -Dtorx=true
   mesa $ meson compile -C build

Running the Test Suite
----------------------

The Torx test suite validates individual neural-network operations against CPU
references.  The models used for this testing are fetched from a separate
repository: https://gitlab.freedesktop.org/tomeu/npu-model-zoo .  At test
time, ``test_torx`` runs directly on the NPU board, executes each ``.pte`` on
the NPU, and compares the result against the CPU reference.

**Prerequisites:**

- Mesa built with ``-Dtorx=true -Dbuild-tests=true``

.. code-block:: console

   mesa $ meson configure build -Dbuild-tests=true
   mesa $ meson compile -C build

**Run:**

.. code-block:: console

   mesa $ TORX_TEST_DATA=build/src/gallium/targets/torx/tests \
          build/src/gallium/targets/torx/test_torx

To run a single test case:

.. code-block:: console

   mesa $ ... build/src/gallium/targets/torx/test_torx --gtest_filter='mobilenet_v2.042'

**Environment variables:**

``TORX_TEST_DATA``
   Directory containing pre-built test artefacts. Defaults to
   ``src/gallium/targets/torx/tests`` under the current directory.

Bisecting a whole-model failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``split_torx_tests.py --ranges`` splits a model into cumulative prefix
subgraphs [0..0] .. [0..N-1] instead of individual ops. Compiling and
running these with the same pipeline lets a whole-model accuracy
failure be binary-searched down to the op that introduces it, while
reproducing the real partitioning decisions for every prefix.

Adding a New Model
~~~~~~~~~~~~~~~~~~

See the `NPU Model Zoo <https://gitlab.freedesktop.org/tomeu/npu-model-zoo>`_
for instructions on adding a new model to the test suite.
