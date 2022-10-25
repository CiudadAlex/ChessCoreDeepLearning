package org.leviatan.chess.engine.intel.deeplearning;

import java.io.IOException;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.engine.CPUPlayer;
import org.leviatan.chess.engine.intel.deeplearning.management.ManagerGeneral;
import org.leviatan.chess.ui.UserIntefaceInteractor;

/**
 * CPUPlayerDeepLearningImpl.
 *
 * @author Alejandro
 *
 */
public class CPUPlayerDeepLearningImpl implements CPUPlayer {

    private final ManagerGeneral managerGeneral;

    /**
     * Constructor for CPUPlayerDeepLearningImpl.
     *
     * @throws IOException
     */
    public CPUPlayerDeepLearningImpl() throws IOException {
        this.managerGeneral = new ManagerGeneral(true);
    }

    @Override
    public Tablero realizarJugadaCPU(final Tablero tablero, final UserIntefaceInteractor userIntefaceInteractor, final Bando bandoCPU) {

        final Movimiento movimiento = this.managerGeneral.calcularMovimiento(tablero, bandoCPU);
        tablero.realizarMovimiento(movimiento);
        return tablero;
    }

}
