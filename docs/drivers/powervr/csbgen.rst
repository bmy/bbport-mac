CSBGEN - Command Stream Buffer Generator
=========================================

CSBGEN is a code generator for pack/unpack functions used by the Imagination
PowerVR GPU drivers. It generates C structures and functions from XML
descriptions of hardware registers and command buffers.

The generator parses XML files that describe the hardware interface and
generates:

- C structures matching hardware register layouts
- Pack functions to convert from C structures to raw binary data for GPU
- Unpack functions to convert from raw binary data back to C structures
- Helper macros and default values for fields

Hardware Interfaces
-------------------

CSBGEN generates code for the following PowerVR Rogue hardware interfaces:

- **CDM** (Compute Data Master) - Compute control stream
- **CR** (Control Registers)
- **IPF** (Internal Parameter Format)
- **LLS** (Low-Latency Scheduling)
- **PBESTATE** (Pixel Backend State)
- **PDS** (Programmable Data Sequencer)
- **PPP** (Primitive Processing Pipeline)
- **TEXSTATE** (Texture State)
- **VDM** (Vertex Data Master) - Geometry control stream


CDM (Compute Data Master)
~~~~~~~~~~~~~~~~~~~~~~~~~

Complete API documentation for CDM control stream structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/cdm.h

CR (Control Registers)
~~~~~~~~~~~~~~~~~~~~~~

API documentation for Control Register structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/cr.h

IPF (Internal Parameter Format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Internal Parameter Format structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/ipf.h

LLS (Low-Latency Scheduling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Low-Latency Scheduling defines:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/lls.h

PBESTATE (Pixel Backend State)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Pixel Backend structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/pbestate.h

PDS (Programmable Data Sequencer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Programmable Data Sequencer structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/pds.h

PPP (Primitive Processing Pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Primitive Processing Pipeline structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/ppp.h

TEXSTATE (Texture State)
~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Texture State structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/texstate.h

VDM (Vertex Data Master)
~~~~~~~~~~~~~~~~~~~~~~~~

API documentation for Vertex Data Master structures and functions:

.. c:autodoc:: @MESA_BUILD_ROOT@/src/imagination/csbgen/rogue/vdm.h

Implementation
--------------

The generator is implemented in Python and located at
:file:`src/imagination/csbgen/gen_pack_header.py`. XML specifications are
located in :file:`src/imagination/csbgen/rogue/`.
