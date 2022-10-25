package org.leviatan.chess.engine.intel.deeplearning.management;

import java.io.IOException;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.BishopRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.KingRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.KnightRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.PawnRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.QueenRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.RookRawNetwork;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.WhichPieceRawNetwork;
import org.leviatan.chess.tools.platform.AppLogger;

/**
 * ManagerBando.
 *
 * @author Alejandro
 *
 */
public class ManagerBando {

    private final WhichPieceRawNetwork whichPieceRawNetwork;
    private final BishopRawNetwork bishopRawNetwork;
    private final KingRawNetwork kingRawNetwork;
    private final KnightRawNetwork knightRawNetwork;
    private final PawnRawNetwork pawnRawNetwork;
    private final QueenRawNetwork queenRawNetwork;
    private final RookRawNetwork rookRawNetwork;

    /**
     * Maneja un bando.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathDirStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public ManagerBando(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {

        this.whichPieceRawNetwork = new WhichPieceRawNetwork(bando, pathDirStoreLoad, load);
        this.bishopRawNetwork = new BishopRawNetwork(bando, pathDirStoreLoad, load);
        this.kingRawNetwork = new KingRawNetwork(bando, pathDirStoreLoad, load);
        this.knightRawNetwork = new KnightRawNetwork(bando, pathDirStoreLoad, load);
        this.pawnRawNetwork = new PawnRawNetwork(bando, pathDirStoreLoad, load);
        this.queenRawNetwork = new QueenRawNetwork(bando, pathDirStoreLoad, load);
        this.rookRawNetwork = new RookRawNetwork(bando, pathDirStoreLoad, load);
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
    public void train(final int numEpochs, final List<Partida> listPartida) throws IOException {

        this.whichPieceRawNetwork.train(numEpochs, listPartida);
        this.bishopRawNetwork.train(numEpochs, listPartida);
        this.kingRawNetwork.train(numEpochs, listPartida);
        this.knightRawNetwork.train(numEpochs, listPartida);
        this.pawnRawNetwork.train(numEpochs, listPartida);
        this.queenRawNetwork.train(numEpochs, listPartida);
        this.rookRawNetwork.train(numEpochs, listPartida);
    }

    /**
     * Calcula el movimiento a realizar.
     *
     * @param tablero
     *            tablero
     * @return el movimiento a realizar
     */
    public Movimiento calcularMovimiento(final Tablero tablero) {

        final List<TipoFicha> listTipoFicha = this.whichPieceRawNetwork.calcularTipoFicha(tablero);

        AppLogger.logDebug("_______________________________________________");

        for (final TipoFicha tipoFicha : listTipoFicha) {

            AppLogger.logDebug("Probando tipo ficha " + tipoFicha.name());

            final Movimiento movimiento = calculaMovimientoPorTipoDeFicha(tablero, tipoFicha);

            if (movimiento != null) {
                return movimiento;
            }
        }

        return null;
    }

    private Movimiento calculaMovimientoPorTipoDeFicha(final Tablero tablero, final TipoFicha tipoFicha) {

        if (TipoFicha.ALFIL.equals(tipoFicha)) {
            return this.bishopRawNetwork.calcularMovimiento(tablero);

        } else if (TipoFicha.CABALLO.equals(tipoFicha)) {
            return this.knightRawNetwork.calcularMovimiento(tablero);

        } else if (TipoFicha.PEON.equals(tipoFicha)) {
            return this.pawnRawNetwork.calcularMovimiento(tablero);

        } else if (TipoFicha.REINA.equals(tipoFicha)) {
            return this.queenRawNetwork.calcularMovimiento(tablero);

        } else if (TipoFicha.REY.equals(tipoFicha)) {
            return this.kingRawNetwork.calcularMovimiento(tablero);

        } else if (TipoFicha.TORRE.equals(tipoFicha)) {
            return this.rookRawNetwork.calcularMovimiento(tablero);
        }

        return null;
    }
}
