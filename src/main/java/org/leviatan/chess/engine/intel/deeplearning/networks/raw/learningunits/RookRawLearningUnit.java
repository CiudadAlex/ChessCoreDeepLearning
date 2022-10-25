package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Ficha;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.movimientosposibles.Direccion;

/**
 * RookRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public class RookRawLearningUnit extends AbstractRawLearningUnit {

    /** NUM_DIRECCIONES. */
    public static final int NUM_DIRECCIONES = 4;

    /** NUM_INTENSIDADES. */
    public static final int NUM_INTENSIDADES = 7;

    /** NUM_OUTPUTS = Dirección + intensidad + num fichas. */
    public static final int NUM_OUTPUTS = NUM_DIRECCIONES + NUM_INTENSIDADES + 2;

    /**
     * Constructor for RookRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public RookRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        super(tablero, movimiento, bando);
    }

    @Override
    public double[] getOutput() {

        final double[] output = new double[NUM_OUTPUTS];

        final Ficha ficha = checkFicha(TipoFicha.TORRE);

        final PosicionTablero posicionTableroOrigen = this.movimiento.getPosicionOrigen();
        final PosicionTablero posicionTableroDestino = this.movimiento.getPosicionDestino();

        final List<Direccion> listDireccion = Direccion.getSubconjuntoDirecciones(false);

        fillInicioOutputConDireccionYIntensidad(output, posicionTableroOrigen, posicionTableroDestino, listDireccion);

        final Integer indexFicha = getIndexFicha(ficha, posicionTableroOrigen);
        output[NUM_DIRECCIONES + NUM_INTENSIDADES + indexFicha] = 1;

        return output;
    }

}
