# Copyright 2026 The Android Open Source Project
# SPDX-License-Identifier: MIT

from .wrapperdefs import VulkanWrapperGenerator
from .common.codegen import CodeGen

import cerealgenerator

class VulkanSupportedExtensions(VulkanWrapperGenerator):
    def __init__(self, module, typeInfo):
        super().__init__(module, typeInfo)
        self.codegen = CodeGen()

    def onBegin(self):
        super().onBegin()
        self.module.appendHeader("""

const std::unordered_set<std::string>&
GetInstanceExtensionsSupportedByCodegen();

const std::unordered_set<std::string>&
GetDeviceExtensionsSupportedByCodegen();

""")

        self.module.appendImpl("const std::unordered_set<std::string>&\n")
        self.module.appendImpl("GetInstanceExtensionsSupportedByCodegen() {\n")
        self.module.appendImpl("    static const std::unordered_set<std::string>* kSupported = []() {\n")
        self.module.appendImpl("        return new std::unordered_set<std::string>{\n")

        for ext in sorted(cerealgenerator.SUPPORTED_INSTANCE_EXTENSIONS):
            self.module.appendImpl(f'            "{ext}",\n')

        self.module.appendImpl("        };\n")
        self.module.appendImpl("    }();\n")
        self.module.appendImpl("    return *kSupported;\n")
        self.module.appendImpl("}\n\n")

        self.module.appendImpl("const std::unordered_set<std::string>&\n")
        self.module.appendImpl("GetDeviceExtensionsSupportedByCodegen() {\n")
        self.module.appendImpl("    static const std::unordered_set<std::string>* kSupported = []() {\n")
        self.module.appendImpl("        return new std::unordered_set<std::string>{\n")

        for ext in sorted(cerealgenerator.SUPPORTED_DEVICE_EXTENSIONS):
            self.module.appendImpl(f'            "{ext}",\n')

        self.module.appendImpl("        };\n")
        self.module.appendImpl("    }();\n")
        self.module.appendImpl("    return *kSupported;\n")
        self.module.appendImpl("}\n")
        self.module.appendImpl("\n")
