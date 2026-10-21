AMD GPU Image2D Tiling Layouts
==============================

These are 2D address bit swizzles with Morton-like interleaving and XOR-based tile flipping.

The bit swizzle notation used here is vendor-agnostic and HW-agnostic. A similar table can be produced for any HW from any vendor.

Any device can support any of these layouts in their image copy shaders by computing the image address using
the bit swizzles below and using plain buffer loads/stores.

If a device supports the same bit swizzle as another device, it means that it supports the exact same tiling layout.

If a device supports the same bit swizzle as another device except having additional terms {Xi, Xi+1, ..., Xi+n-1}
at the end, it means that it supports the exact same tiling layout except that its pitch alignment is also multiplied
by 2^n compared to the other device. That means the other device supports the exact same tiling layout if it increases
its pitch alignment accordingly.

The table doesn't list all existing chips. We can add more chips on request.

Automatically generated.

.. csv-table::
    :header: "Chip"| "Bpp"| "HW mode"| "Tile dim."| "2D address bit swizzle"
    :file: tiling-layouts.csv
    :widths: 10, 8, 7, 8, 50
    :delim: |
