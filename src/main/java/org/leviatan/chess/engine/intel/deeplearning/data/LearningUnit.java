package org.leviatan.chess.engine.intel.deeplearning.data;

/**
 * LearningUnit.
 *
 * @author Alejandro
 *
 */
public interface LearningUnit {

    /**
     * Devuelve el input.
     *
     * @return INDArray
     */
    public double[] getInput();

    /**
     * Devuelve el output.
     *
     * @return INDArray
     */
    public double[] getOutput();

    /**
     * Devuelve el LearningData.
     *
     * @return LearningData
     */
    default LearningData buildLearningData() {

        try {
            return new LearningData(getInput(), getOutput());

        } catch (final CheckFichaRuntimeException e) {
            // Do nothing
        }

        return null;
    }
}
