package org.leviatan.chess.engine.intel.deeplearning.data;

/**
 * LearningData.
 *
 * Hay que calcular arrays inmediatamente para evitar problemas de modificacion
 * del tablero
 *
 * @author Alejandro
 *
 */
public class LearningData {

    private final double[] input;
    private final double[] output;

    /**
     * Constructor of LearningData.
     *
     * @param input
     *            input
     * @param output
     *            output
     */
    public LearningData(final double[] input, final double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return this.input;
    }

    public double[] getOutput() {
        return this.output;
    }

}
