package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Ficha;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroEnL;

/**
 * KnightRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public class KnightRawLearningUnit extends AbstractRawLearningUnit {

    /** NUM_MOVIMIENTOS. */
    public static final int NUM_MOVIMIENTOS = 8;

    /** NUM_OUTPUTS = movimientos + num fichas. */
    public static final int NUM_OUTPUTS = NUM_MOVIMIENTOS + 2;

    /**
     * Constructor for KnightRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public KnightRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        super(tablero, movimiento, bando);
    }

    @Override
    public double[] getOutput() {

        final double[] output = new double[NUM_OUTPUTS];

        final Ficha ficha = checkFicha(TipoFicha.CABALLO);

        final PosicionTablero posicionTableroOrigen = this.movimiento.getPosicionOrigen();
        final PosicionTablero posicionTableroDestino = this.movimiento.getPosicionDestino();

        final List<PosicionTablero> listPosicionTableroEnL = RepositorioPosicionesTableroEnL
                .getListaPosicionTableroPosicionesEnL(posicionTableroOrigen.getHorizontal(), posicionTableroOrigen.getVertical());

        int index = 0;

        for (final PosicionTablero posicionTableroEnL : listPosicionTableroEnL) {

            if (posicionTableroEnL.equals(posicionTableroDestino)) {
                output[index] = 1;
            }

            index++;
        }

        final Integer indexFicha = getIndexFicha(ficha, posicionTableroOrigen);
        output[NUM_MOVIMIENTOS + indexFicha] = 1;

        return output;

    }
}
