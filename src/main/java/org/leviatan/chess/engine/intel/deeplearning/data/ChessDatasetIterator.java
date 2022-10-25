package org.leviatan.chess.engine.intel.deeplearning.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.TimeProgressLogger;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ChessDatasetIterator<T extends LearningUnit> implements DataSetIterator {

    private static final long serialVersionUID = 1655984149540148845L;

    private final List<Partida> listPartida;

    private Iterator<Partida> iterPartida;
    private Iterator<LearningData> iterLearningData;

    private final int numInputs;
    private final int numOutputs;
    private final Function3Arg<Tablero, Movimiento, Bando, T> builderLearningUnit;
    private final Bando bando;
    private TimeProgressLogger progress;

    /**
     * Constuctor for SingleOutputDenseDatasetIterator.
     *
     * @param listPartida
     *            listPartida
     * @param numInputs
     *            numInputs
     * @param numOutputs
     *            numOutputs
     * @param builderLearningUnit
     *            builderLearningUnit
     * @param bando
     *            bando
     */
    public ChessDatasetIterator(final List<Partida> listPartida, final int numInputs, final int numOutputs,
            final Function3Arg<Tablero, Movimiento, Bando, T> builderLearningUnit, final Bando bando) {

        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.builderLearningUnit = builderLearningUnit;
        this.bando = bando;

        this.listPartida = listPartida;
        this.iterPartida = listPartida.iterator();
        this.progress = new TimeProgressLogger(listPartida.size(), 100);
    }

    @Override
    public boolean hasNext() {
        return this.iterLearningData != null && this.iterLearningData.hasNext() || this.iterPartida.hasNext();
    }

    @Override
    public DataSet next() {
        return next(1);
    }

    @Override
    public DataSet next(final int num) {

        final List<LearningData> listLearningData = new ArrayList<LearningData>();

        for (int i = 0; i < num; i++) {

            final LearningData learningData = getNextLearningData();

            if (learningData != null) {
                listLearningData.add(learningData);
            } else {
                break;
            }
        }

        final int currMinibatchSize = listLearningData.size();

        if (currMinibatchSize == 0) {
            return null;
        }

        return DeepLearningUtils.buildDataSet(listLearningData, ld -> ld.getInput(), ld -> ld.getOutput(), this.numInputs, this.numOutputs);
    }

    private LearningData getNextLearningData() {

        if (this.iterLearningData != null && this.iterLearningData.hasNext()) {
            return this.iterLearningData.next();
        }

        if (this.iterPartida.hasNext()) {

            final Partida partida = this.iterPartida.next();
            this.iterLearningData = LearningDataIteratorBuilder.build(partida, this.builderLearningUnit, this.bando);

            this.progress.stepFinishedAndPrintProgress();

            if (this.iterLearningData.hasNext()) {
                return this.iterLearningData.next();
            }
        }

        return null;
    }

    @Override
    public int inputColumns() {
        return this.numInputs;
    }

    @Override
    public int totalOutcomes() {
        return this.numOutputs;
    }

    @Override
    public void reset() {
        this.iterPartida = this.listPartida.iterator();
        this.iterLearningData = null;
        this.progress = new TimeProgressLogger(this.listPartida.size(), 100);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return 1;
    }

    @Override
    public void setPreProcessor(final DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not implemented");
    }

}
