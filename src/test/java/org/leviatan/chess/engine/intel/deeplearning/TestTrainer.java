package org.leviatan.chess.engine.intel.deeplearning;

import java.util.List;

import org.leviatan.chess.data.pgn.PGNReaderManager;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.intel.deeplearning.management.ManagerGeneral;
import org.leviatan.chess.tools.platform.AppLogger;

/**
 * TestTrainer.
 *
 * @author Alejandro
 *
 */
public final class TestTrainer {

    private TestTrainer() {
    }

    /**
     * Main method.
     *
     * @param args
     *            args
     * @throws Exception
     */
    public static void main(final String[] args) throws Exception {

        final boolean load = false;
        final ManagerGeneral managerGeneral = new ManagerGeneral(load);

        final int numEpochs = 1;
        final List<Partida> listPartida = PGNReaderManager.getPartidasDelFichero(0);

        AppLogger.logDebug("Entrenando con el numero de partidas: " + listPartida.size());

        managerGeneral.train(numEpochs, listPartida);
    }
}
