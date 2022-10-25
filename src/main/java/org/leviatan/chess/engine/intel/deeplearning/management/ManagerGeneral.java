package org.leviatan.chess.engine.intel.deeplearning.management;

import java.io.IOException;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.deeplearning.ConfiguracionDeepLearning;
import org.leviatan.chess.tools.platform.AppLogger;

/**
 * ManagerGeneral.
 *
 * @author Alejandro
 *
 */
public class ManagerGeneral {

    private final ManagerBando managerBandoBlanco;
    private final ManagerBando managerBandoNegro;

    /**
     * Constructor for ManagerGeneral.
     *
     * @param load
     *            load
     * @throws IOException
     */
    public ManagerGeneral(final boolean load) throws IOException {

        final String pathDirStoreLoad = ConfiguracionDeepLearning.DIR_MODELOS;

        this.managerBandoBlanco = new ManagerBando(Bando.BLANCO, pathDirStoreLoad, load);
        this.managerBandoNegro = new ManagerBando(Bando.NEGRO, pathDirStoreLoad, load);
    }

    /**
     * Entrena las redes.
     *
     * @param numEpochs
     *            numEpochs
     * @param listPartida
     *            listPartida
     * @throws IOException
     */
    public void train(final int numEpochs, final List<Partida> listPartida) throws Exception {

        this.managerBandoBlanco.train(numEpochs, listPartida);
        this.managerBandoNegro.train(numEpochs, listPartida);
    }

    /**
     * Calcula el movimiento a realizar.
     *
     * @param tablero
     *            tablero
     * @param bando
     *            bando
     * @return el movimiento a realizar
     */
    public Movimiento calcularMovimiento(final Tablero tablero, final Bando bando) {

        if (Bando.BLANCO.equals(bando)) {
            AppLogger.logDebug("Calculando movimiento BLANCO");
            return this.managerBandoBlanco.calcularMovimiento(tablero);

        } else if (Bando.NEGRO.equals(bando)) {
            AppLogger.logDebug("Calculando movimiento NEGRO");
            return this.managerBandoNegro.calcularMovimiento(tablero);

        }
        return null;
    }
}
