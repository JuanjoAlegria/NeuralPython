import numpy as np
import Hyperparameters
import matplotlib.pyplot as plt
from NeuralPython.Network import Network


def rebuildSignal(listModelDirs, data, listErrors = []):
    hpDict = Hyperparameters.buildFromArgs()
    networks = []
    for d in listModelDirs:
        network = Network.buildFromDict(hpDict)
        dirName = "NetworksModels/" + d
        network.load(dirName)
        networks.append(network)

    outputsTotal = []
    xs, xes, ys = data
    errorsC1 = []
    errorsC2 = []
    for network in networks:
        outputs = np.empty(0)
        e1 = 0.0
        e2 = 0.0
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            xE = xes[i] if xes != [] else []

            output = network.forwardOneSample(x, xE)
            outputs = np.append(outputs, output)

            e1 += (y - output)**2
            e2 += (y - output)**2 / (y**2)

        errorsC1.append(e1 / (2.0 * len(xs)))
        errorsC2.append(e2 / (2.0 * len(xs)))
        outputs = np.array(outputs).astype(float)
        outputsTotal.append(outputs)

    return outputsTotal, errorsC1, errorsC2

def plotSignal(data, outputs, labels = [], title = "", xlabel = "", ylabel = ""):
    xs, xes, ys = data
    ts = range(len(outputs[0]))
    if labels == []:
        labels = ["Datos experimentales " + str(i) for i in range(len(outputs))]

    experimental = []
    for i in range(len(outputs)):
        e, = plt.plot(ts, outputs[i], label = labels[i], marker='*')
        experimental.append(e)
    reals, = plt.plot(ts, ys + xs[:, :, -1], label='Datos reales', marker='o')
    labels = experimental[:]
    labels.append(reals)
    plt.legend(handles=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def rebuildFilters(filename, rows = 4, columns = 3, title = "Filtros obtenidos", \
                   xlabel = 'Tiempo', ylabel = 'Magnitud de los filtros'):
    filters = np.load(filename)
    f, axarr = plt.subplots(rows, columns, sharex='col', sharey='row')

    for i in range(rows):
        for j in range(columns):
            n = columns*i + j
            m = filters[0,n]
            axarr[i,j].plot(m)
            axarr[i,j].autoscale(enable = True, axis = 'x', tight = True)


    axarr[0,(columns - 1) / 2].set_title(title)
    axarr[(rows - 1) / 2,0].set_ylabel(ylabel)
    axarr[(rows - 1), (columns - 1) / 2].set_xlabel(xlabel)
    plt.show()


def concatErrorProgression(listErrorFiles):
    # se asume que la lista de errores está ordenada
    files = [open(f) for f in listErrorFiles]
    errors = []

    for f in files:
        errorsIn = []
        errorsOut = []
        for line in f:
            if "Error IN" in line:
                error = float(line[line.index("[") + 2 : line.index("]")])
                errorsIn.append(error)
            elif "Error OUT" in line:
                error = float(line[line.index("[") + 2 : line.index("]")])
                errorsOut.append(error)
        errors.append((errorsIn, errorsOut))


    lastError = errors[0][0][-1]
    totalErrorsIn = errors[0][0]
    totalErrorsOut = errors[0][1]
    for j in range(1, len(errors)):
        currentErrorIn = errors[j][0]
        currentErrorOut = errors[j][1]
        index = 0
        for i in range(len(currentErrorIn)):
            e = currentErrorIn[i]
            if e < lastError:
                index = i
                break
        totalErrorsIn += currentErrorIn[index:]
        totalErrorsOut += errorsOut + currentErrorOut[index:]

        lastError = currentErrorIn[-1]

    return np.array(totalErrorsIn), np.array(totalErrorsOut)

def plotErrorProgression(errorsIn, errorsOut, logX = True, logY = True):

    epochs = range(len(errorsIn))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    inPlot, = ax.plot(epochs, errorsIn, label='Error dentro de la muestra')
    outPlot, = ax.plot(epochs, errorsOut, label='Error fuera de la muestra')
    ax.legend(handles=[inPlot, outPlot])

    xlabel = "Epoch"
    ylabel = "Error obtenido"
    if logX:
        ax.set_xscale('log')
        xlabel += " (escala logarítmica)"
    if logY:
        ax.set_yscale('log')
        ylabel += " (escala logarítmica)"

    plt.title("Error dentro y fuera de la muestra de entrenamiento vs epochs")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
