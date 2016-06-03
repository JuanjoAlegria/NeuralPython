# -*- coding: utf-8 -*-
import argparse
import json
import os

baseFileName = os.path.dirname(os.path.realpath(__file__)) + "/../Config/{0}.json"

def loadHyperParameters(jsonFile=None):
    defaultDict = json.load(open(baseFileName.format("defaultHyperParameters")))
    if jsonFile == None:
        return defaultDict
    # merge de diccionarios
    hpDict = json.load(open(baseFileName.format(jsonFile)))
    newDict = defaultDict.copy()
    newDict.update(hpDict)
    return newDict

def updateDict(oldDict, newDict):
    d = oldDict.copy()
    d.update(newDict)
    return d

def showHyperParameters(d):
    for key in sorted(d.keys()):
        print key, ":", d[key]

def saveToFile(d, filename):
    json.dump(d, open(baseFileName.format(filename), "w"))

def modifyHyperParameters(d):
    newDict = {}
    while True:
        key = raw_input('Ingrese llave que desea modificar (0 para terminar, SHOW para mostrar las opciones posibles): ')
        if key == "0":
            break
        if key == "SHOW":
            showHyperParameters(d)
        if key not in d:
            print "Error, la llave ingresada no existe, trate nuevamente"
            continue
        value = raw_input('Ingrese nuevo valor para la llave ' + key + ": ")
        newDict[key] = value
    print "Llaves modificadas:"
    for key in newDict:
        print key
        print "Antiguo valor: ", d[key], "Nuevo valor:", newDict[key]

    save = raw_input("¿Desea guardar en archivo la nueva configuración? (s / n) : ")
    if save == "s":
        fileName = raw_input("Ingrese nombre de archivo a guardar: " )
        saveToFile(newDict, fileName)
    return newDict

def buildFromArgs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-s", "--show", action="store_const", const = True, default = False,
                        help = "Muestra los hiperparámetros actuales")
    parser.add_argument("--cfgFile", metavar = 'file', type = str,
                        default = None, help = "Archivo JSON con hiperparámetros de configuración")
    parser.add_argument("-m", "--modify", action = "store_const", const = True,
                        default = False, help = "Modo interactivo para modificar hiperparámetros")

    args = parser.parse_args()
    hp = loadHyperParameters(args.cfgFile)
    if args.show:
        showHyperParameters(hp)

    if args.modify:
        modifications = modifyHyperParameters(hp)
        finalDict = updateDict(hp, modifications)
    else:
        finalDict = hp

    return finalDict
