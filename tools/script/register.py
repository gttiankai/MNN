#!/usr/bin/python
import os
def generateShape(rootDir):
    shapeDir = os.path.join(rootDir, "source", "shape")
    shapeRegFile = os.path.join(shapeDir, "ShapeRegister.cpp")
    print(shapeRegFile)
    fileNames = os.listdir(shapeDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    shapeLists = []
    for fi in fileNames:
        if ".hpp" in fi:
            continue
        f = os.path.join(shapeDir, fi)
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_SHAPE')>=0, c))
            for l in c:
                if l.find('REGISTER_SHAPE(')>=0:
                    l = l.replace("REGISTER_SHAPE(", "")
                    l = l.split(')')[0]
                    l = l.replace(' ', "")
                    l = l.split(',')
                    func = '___' + l[0] + '__'+l[1]+"__"
                    shapeLists.append(func)
                elif l.find('REGISTER_SHAPE_OLD(')>=0:
                    l = l.replace("REGISTER_SHAPE_OLD(", "")
                    l = l.split(')')[0]
                    l = l.replace(' ', "")
                    l = l.split(',')
                    func = '___' + l[0] + '__'+l[1]+"__"
                    shapeLists.append(func)
                elif l.find('REGISTER_SHAPE_INPUTS(') >= 0:
                    l = l.replace("REGISTER_SHAPE_INPUTS(", "")
                    l = l.split(')')[0]
                    l = l.replace(' ', "")
                    l = l.split(',')
                    func = '___' + l[0] + '__'+l[1]+"__"
                    shapeLists.append(func)
    with open(shapeRegFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('namespace MNN {\n')
        for l in shapeLists:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerShapeOps() {\n')
        for l in shapeLists:
            f.write(l+'();\n')
        f.write("}\n}\n")
    return

def generateCPUFile(rootDir):
    cpuDir = os.path.join(rootDir, "source", "backend", "cpu")
    cpuRegFile = os.path.join(cpuDir, "CPUOPRegister.cpp")
    fileNames = os.listdir(cpuDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    funcNames = []
    for fi in fileNames:
        f = os.path.join(cpuDir, fi)
        if os.path.isdir(f):
            continue
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_CPU_OP_CREATOR')>=0, c))
            c = list(filter(lambda l:l.find('OpType')>=0, c))
            for l in c:
                l = l.split('(')[1]
                l = l.split(')')[0]
                l = l.replace(' ', '')
                l = l.split(',')
                funcName = '___' + l[0] + '__' + l[1] + '__'
                funcNames.append(funcName)
    with open(cpuRegFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('namespace MNN {\n')
        for l in funcNames:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerCPUOps() {\n')
        for l in funcNames:
            f.write(l+'();\n')
        f.write("}\n}\n")

def generateGeoFile(rootDir):
    geoDir = os.path.join(rootDir, "source", "geometry")
    regFile = os.path.join(geoDir, "GeometryOPRegister.cpp")
    fileNames = os.listdir(geoDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    funcNames = []
    for fi in fileNames:
        if ".hpp" in fi:
            continue
        f = os.path.join(geoDir, fi)
        if os.path.isdir(f):
            continue
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_GEOMETRY')>=0, c))
            for l in c:
                l = l.split('(')[1]
                l = l.split(')')[0]
                l = l.replace(' ', '')
                l = l.split(',')
                funcName = '___' + l[0] + '__' + l[1] + '__'
                funcNames.append(funcName)

    with open(regFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('#include \"geometry/GeometryComputer.hpp\"\n')
        f.write('namespace MNN {\n')
        for l in funcNames:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerGeometryOps() {\n')
        for l in funcNames:
            f.write(l+'();\n')
        f.write("}\n}\n")

def generateCoreMLFile(rootDir):
    coremlDir = os.path.join(rootDir, "source", "backend", "coreml")
    coremlExeDir = os.path.join(coremlDir, "execution")
    coremlRegFile = os.path.join(coremlDir, "backend", "CoreMLOPRegister.cpp")
    fileNames = os.listdir(coremlExeDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    funcNames = []
    for fi in fileNames:
        f = os.path.join(coremlExeDir, fi)
        if os.path.isdir(f):
            continue
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_COREML_OP_CREATOR')>=0, c))
            c = list(filter(lambda l:l.find('OpType')>=0, c))
            for l in c:
                l = l.split('(')[1]
                l = l.split(')')[0]
                l = l.replace(' ', '')
                l = l.split(',')
                funcName = '___' + l[0] + '__' + l[1] + '__'
                funcNames.append(funcName)
    with open(coremlRegFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('namespace MNN {\n')
        for l in funcNames:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerCoreMLOps() {\n')
        for l in funcNames:
            f.write(l+'();\n')
        f.write("}\n}\n")

def generateNNAPIFile(rootDir):
    coremlDir = os.path.join(rootDir, "source", "backend", "nnapi")
    coremlExeDir = os.path.join(coremlDir, "execution")
    coremlRegFile = os.path.join(coremlDir, "backend", "NNAPIOPRegister.cpp")
    fileNames = os.listdir(coremlExeDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    funcNames = []
    for fi in fileNames:
        f = os.path.join(coremlExeDir, fi)
        if os.path.isdir(f):
            continue
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_NNAPI_OP_CREATOR')>=0, c))
            c = list(filter(lambda l:l.find('OpType')>=0, c))
            for l in c:
                l = l.split('(')[1]
                l = l.split(')')[0]
                l = l.replace(' ', '')
                l = l.split(',')
                funcName = '___' + l[0] + '__' + l[1] + '__'
                funcNames.append(funcName)
    with open(coremlRegFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('namespace MNN {\n')
        for l in funcNames:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerNNAPIOps() {\n')
        for l in funcNames:
            f.write(l+'();\n')
        f.write("}\n}\n")

import sys
generateShape(sys.argv[1])
generateCPUFile(sys.argv[1])
generateGeoFile(sys.argv[1])
generateCoreMLFile(sys.argv[1])
generateNNAPIFile(sys.argv[1])
