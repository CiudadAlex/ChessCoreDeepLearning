package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Ficha;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;

/**
 * WhichPieceRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public class WhichPieceRawLearningUnit extends AbstractRawLearningUnit {

    /** NUM_OUTPUTS = Numero de tipos de ficha. */
    public static final int NUM_OUTPUTS = TipoFicha.values().length;

    /**
     * Constructor for WhichPieceRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public WhichPieceRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        super(tablero, movimiento, bando);
    }

    @Override
    public double[] getOutput() {

        final Ficha ficha = this.tablero.getFicha(this.movimiento.getPosicionOrigen());
        final TipoFicha tipoFicha = ficha.getTipoFicha();

        final TipoFicha[] arrayTipoFicha = TipoFicha.values();

        final double[] output = new double[arrayTipoFicha.length];

        for (int i = 0; i < arrayTipoFicha.length; i++) {

            if (tipoFicha.equals(arrayTipoFicha[i])) {
                output[i] = 1;
            }
        }

        return output;
    }
}
