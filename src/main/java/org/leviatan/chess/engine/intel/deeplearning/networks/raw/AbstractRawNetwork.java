package org.leviatan.chess.engine.intel.deeplearning.networks.raw;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Casilla;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.intel.deeplearning.data.ChessDatasetIterator;
import org.leviatan.chess.engine.intel.deeplearning.data.LearningUnit;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits.AbstractRawLearningUnit;
import org.leviatan.chess.engine.movimientosposibles.Direccion;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroDireccion;
import org.leviatan.chess.tools.platform.AppLogger;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.KeyDoubleBean;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * AbstractRawNetwork.
 *
 * Deep learning: En chess hacer una red neuronal a la que se le pasa en tablero
 * entero y que elija el tipo de ficha a mover y otras 6 redes neuronales que
 * decidan la posición en orden de las casillas del tipo de ficha y el
 * movimiento concreto por tipo de ficha pasandoles el tablero entero y la
 * posicion de la ficha a mover
 *
 * @author Alejandro
 *
 */
public abstract class AbstractRawNetwork<T extends LearningUnit> {

    protected final Bando bando;
    protected final String pathDirStoreLoad;
    protected final boolean load;

    protected final MultiLayerNetwork net;

    /**
     * Constructor for AbstractRawNetwork.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public AbstractRawNetwork(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {
        this.bando = bando;
        this.pathDirStoreLoad = pathDirStoreLoad;
        this.load = load;

        if (load) {

            final String pathModel = getPathModel();
            this.net = ModelSerializer.restoreMultiLayerNetwork(pathModel, true);

        } else {

            final MultiLayerConfiguration conf = buildNetwork();

            this.net = new MultiLayerNetwork(conf);
            this.net.init();
        }

        this.net.setListeners(new ScoreIterationListener(100));
    }

    /**
     * Entrena la red.
     *
     * @param numEpochs
     *            numEpochs
     * @param listPartida
     *            listPartida
     * @throws IOException
     */
    public void train(final int numEpochs, final List<Partida> listPartida) throws IOException {

        final int numInputs = AbstractRawLearningUnit.NUM_INPUTS_TABLERO;
        final int numOutputs = getNumOutputs();
        final Function3Arg<Tablero, Movimiento, Bando, T> builderLearningUnit = getBuilderLearningUnit();

        AppLogger.logDebug("Entrenando red " + this.getClass().getSimpleName() + " " + this.bando.name());
        final DataSetIterator dataSetIterator = new ChessDatasetIterator<T>(listPartida, numInputs, numOutputs, builderLearningUnit,
                this.bando);

        for (int i = 0; i < numEpochs; i++) {

            this.net.fit(dataSetIterator);
            dataSetIterator.reset();
        }

        final String pathModel = getPathModel();
        ModelSerializer.writeModel(this.net, pathModel, true);
    }

    protected String getPathModel() {
        return this.pathDirStoreLoad + "/" + getModelFileName();
    }

    protected String getModelFileName() {
        return this.getClass().getSimpleName() + "_" + this.bando.name().toLowerCase() + ".model";
    }

    protected abstract MultiLayerConfiguration buildNetwork();

    protected abstract int getNumOutputs();

    protected abstract Function3Arg<Tablero, Movimiento, Bando, T> getBuilderLearningUnit();

    protected MultiLayerConfiguration buildNetwork(final int numHiddenNodes0, final int numHiddenNodes1) {

        final int numInputs = AbstractRawLearningUnit.NUM_INPUTS_TABLERO;
        final int numOutputs = getNumOutputs();

        final int seed = 12345;
        final double learningRate = 0.00001;

        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(learningRate, 0.9)).list()
                .layer(0,
                        new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes0).weightInit(WeightInit.XAVIER)
                                .activation(Activation.RELU).build())
                .layer(1,
                        new DenseLayer.Builder().nIn(numHiddenNodes0).nOut(numHiddenNodes1).weightInit(WeightInit.XAVIER)
                                .activation(Activation.RELU).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).nIn(numHiddenNodes1).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        return conf;
    }

    protected double[] test(final Tablero tablero) {

        final double[] inputByteArray = AbstractRawLearningUnit.getInput(tablero);
        final INDArray input = Nd4j.create(new int[] { 1, inputByteArray.length }, 'f');

        for (int j = 0; j < inputByteArray.length; j++) {
            input.putScalar(new int[] { 0, j }, inputByteArray[j]);
        }

        final int numOutputs = getNumOutputs();
        final double[] outputByteArray = new double[numOutputs];

        final INDArray networkOutput = this.net.output(input);

        for (int i = 0; i < numOutputs; i++) {
            outputByteArray[i] = networkOutput.getDouble(i);
        }

        DeepLearningUtils.printArray(outputByteArray);

        return outputByteArray;
    }

    protected boolean esMovimientoPosible(final Tablero tablero, final PosicionTablero posicionOrigen,
            final PosicionTablero posicionDestino) {

        final TipoFicha tipoFicha = tablero.getCasilla(posicionOrigen).getFicha().getTipoFicha();

        final List<Movimiento> listaMovimientos = new ArrayList<Movimiento>();
        tipoFicha.getGeneradorMovimientosPosibles().generaMovimientosPosiblesYAddToLista(posicionOrigen, tablero, listaMovimientos);

        for (final Movimiento movimiento : listaMovimientos) {

            if (movimiento.getPosicionDestino().equals(posicionDestino)) {
                // Es un movimiento posible
                return true;
            }
        }

        return false;
    }

    protected Movimiento getMovimientoDireccion(final Tablero tablero, final List<KeyDoubleBean<Integer>> listIndexCualFicha,
            final List<KeyDoubleBean<Integer>> listIndexDireccion, final List<KeyDoubleBean<Integer>> listIndexIntensidad,
            final List<Direccion> listDireccion, final List<Casilla> listCasillaFicha, final int indexFichaOffset) {

        for (final KeyDoubleBean<Integer> indexFicha : listIndexCualFicha) {

            final int index = indexFicha.getKey() - indexFichaOffset;
            AppLogger.logDebug("Probando index " + index + " de un tamaño " + listCasillaFicha.size());

            if (index >= listCasillaFicha.size()) {
                continue;
            }

            final PosicionTablero posicionOrigen = listCasillaFicha.get(index).getPosicionTablero();
            AppLogger.logDebug("Probando posicionOrigen " + posicionOrigen);

            final Movimiento movimiento = getMovimientoDireccionParaOrigenDado(tablero, listDireccion, listIndexDireccion,
                    listIndexIntensidad, posicionOrigen);

            if (movimiento != null) {
                return movimiento;
            }
        }

        // No era posible ningun movimiento
        return null;
    }

    private Movimiento getMovimientoDireccionParaOrigenDado(final Tablero tablero, final List<Direccion> listDireccion,
            final List<KeyDoubleBean<Integer>> listIndexDireccion, final List<KeyDoubleBean<Integer>> listIndexIntensidad,
            final PosicionTablero posicionOrigen) {

        for (final KeyDoubleBean<Integer> indexDireccion : listIndexDireccion) {

            final int index = indexDireccion.getKey();
            final Direccion direccion = listDireccion.get(index);

            AppLogger.logDebug("Probando direccion " + direccion);

            final Movimiento movimiento = getMovimientoDireccionParaOrigenDadoYDireccionDada(tablero, listIndexIntensidad, posicionOrigen,
                    direccion);

            if (movimiento != null) {
                return movimiento;
            }
        }

        // No era posible ningun movimiento con el origen dado
        return null;
    }

    private Movimiento getMovimientoDireccionParaOrigenDadoYDireccionDada(final Tablero tablero,
            final List<KeyDoubleBean<Integer>> listIndexIntensidad, final PosicionTablero posicionOrigen, final Direccion direccion) {

        final List<PosicionTablero> listPosicionTableroDireccion = RepositorioPosicionesTableroDireccion
                .getListaPosicionTableroPosicionesDireccion(posicionOrigen.getHorizontal(), posicionOrigen.getVertical(), direccion);

        for (final KeyDoubleBean<Integer> indexIntensidad : listIndexIntensidad) {

            final int index = indexIntensidad.getKey();
            final PosicionTablero posicionDestino = listPosicionTableroDireccion.get(index);

            AppLogger.logDebug("Probando posicionDestino " + posicionDestino);

            final boolean esMovimientoPosible = esMovimientoPosible(tablero, posicionOrigen, posicionDestino);

            if (esMovimientoPosible) {
                return new Movimiento(posicionOrigen, posicionDestino);
            }
        }

        // No era posible ningun movimiento con el origen dado y la direccion
        // dada
        return null;
    }

    protected Movimiento getMovimientoParaOrigenDado(final Tablero tablero, final List<PosicionTablero> listPosicionTableroDestino,
            final List<KeyDoubleBean<Integer>> listIndexPosicionFinal, final PosicionTablero posicionOrigen) {

        for (final KeyDoubleBean<Integer> indexPosicionFinal : listIndexPosicionFinal) {

            final int index = indexPosicionFinal.getKey();
            final PosicionTablero posicionDestino = listPosicionTableroDestino.get(index);

            if (posicionDestino == null) {
                continue;
            }

            AppLogger.logDebug("Probando posicionDestino " + posicionDestino);

            final boolean esMovimientoPosible = esMovimientoPosible(tablero, posicionOrigen, posicionDestino);

            if (esMovimientoPosible) {
                return new Movimiento(posicionOrigen, posicionDestino);
            }
        }

        return null;
    }
}
