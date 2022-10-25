package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroAlrededor;

/**
 * KingRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public class KingRawLearningUnit extends AbstractRawLearningUnit {

    /** NUM_MOVIMIENTOS_ALREDEDOR. */
    public static final int NUM_MOVIMIENTOS_ALREDEDOR = 8;

    /** NUM_OUTPUTS = Direcciones más enroques. */
    public static final int NUM_OUTPUTS = NUM_MOVIMIENTOS_ALREDEDOR + 2;

    /**
     * Constructor for KingRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public KingRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        super(tablero, movimiento, bando);
    }

    @Override
    public double[] getOutput() {

        final double[] output = new double[NUM_OUTPUTS];

        checkFicha(TipoFicha.REY);

        final PosicionTablero posicionTableroOrigen = this.movimiento.getPosicionOrigen();
        final PosicionTablero posicionTableroDestino = this.movimiento.getPosicionDestino();

        final List<PosicionTablero> listaPosicionTableroAlrededor = RepositorioPosicionesTableroAlrededor
                .getListaPosicionTableroPosicionesAlrededor(posicionTableroOrigen.getHorizontal(), posicionTableroOrigen.getVertical());

        int index = 0;
        boolean esAlrededor = false;

        for (final PosicionTablero posicionTableroAlrededor : listaPosicionTableroAlrededor) {

            if (posicionTableroAlrededor.equals(posicionTableroDestino)) {
                output[index] = 1;
                esAlrededor = true;
            }

            index++;
        }

        if (!esAlrededor) {

            if (posicionTableroOrigen.getHorizontal() > posicionTableroDestino.getHorizontal()) {
                // Salto hacia la izquierda
                output[NUM_MOVIMIENTOS_ALREDEDOR] = 1;
            } else {
                // Salto hacia la derecha
                output[NUM_MOVIMIENTOS_ALREDEDOR + 1] = 1;
            }
        }

        return output;
    }
}
