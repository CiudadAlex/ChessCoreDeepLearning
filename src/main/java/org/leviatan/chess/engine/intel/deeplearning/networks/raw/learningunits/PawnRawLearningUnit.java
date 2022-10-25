package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Ficha;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.movimientosposibles.dto.PosicionesPeon;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroPeon;

/**
 * PawnRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public class PawnRawLearningUnit extends AbstractRawLearningUnit {

    /** NUM_MOVIMIENTOS. */
    public static final int NUM_MOVIMIENTOS = 4;

    /** NUM_OUTPUTS = movimientos + num fichas. */
    public static final int NUM_OUTPUTS = NUM_MOVIMIENTOS + 8;

    /**
     * Constructor for PawnRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public PawnRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        super(tablero, movimiento, bando);
    }

    @Override
    public double[] getOutput() {

        final double[] output = new double[NUM_OUTPUTS];

        final Ficha ficha = checkFicha(TipoFicha.PEON);

        final PosicionTablero posicionTableroOrigen = this.movimiento.getPosicionOrigen();
        final PosicionTablero posicionTableroDestino = this.movimiento.getPosicionDestino();

        final PosicionesPeon posicionesPeonFinales = RepositorioPosicionesTableroPeon.getListaPosicionTableroPosicionesPeon(
                posicionTableroOrigen.getHorizontal(), posicionTableroOrigen.getVertical(), this.bando);
        final List<PosicionTablero> listPosicionesTableroFinales = posicionesPeonFinales.getListPosicionTablero();

        int index = 0;

        for (final PosicionTablero posicionTableroFinal : listPosicionesTableroFinales) {

            if (posicionTableroFinal != null && posicionTableroFinal.equals(posicionTableroDestino)) {
                output[index] = 1;
            }

            index++;
        }

        final Integer indexFicha = getIndexFicha(ficha, posicionTableroOrigen);
        output[NUM_MOVIMIENTOS + indexFicha] = 1;

        return output;
    }

}
