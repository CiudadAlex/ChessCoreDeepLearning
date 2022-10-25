package org.leviatan.chess.engine.intel.deeplearning.networks.raw;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Casilla;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits.KnightRawLearningUnit;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.structure.RawNetStructure;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroEnL;
import org.leviatan.chess.tools.platform.AppLogger;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.KeyDoubleBean;

/**
 * KnightRawNetwork.
 *
 * @author Alejandro
 *
 */
public class KnightRawNetwork extends AbstractRawNetwork<KnightRawLearningUnit> {

    /**
     * Constructor for KnightRawNetwork.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathDirStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public KnightRawNetwork(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {
        super(bando, pathDirStoreLoad, load);
    }

    @Override
    protected MultiLayerConfiguration buildNetwork() {

        final int numHiddenNodes0 = RawNetStructure.NUM_HIDDEN_NODES_0;
        final int numHiddenNodes1 = RawNetStructure.NUM_HIDDEN_NODES_1;

        return buildNetwork(numHiddenNodes0, numHiddenNodes1);
    }

    @Override
    protected int getNumOutputs() {
        return KnightRawLearningUnit.NUM_OUTPUTS;
    }

    @Override
    protected Function3Arg<Tablero, Movimiento, Bando, KnightRawLearningUnit> getBuilderLearningUnit() {
        return (tablero, movimiento, bando) -> new KnightRawLearningUnit(tablero, movimiento, bando);
    }

    /**
     * Calcula el movimiento a realizar.
     *
     * @param tablero
     *            tablero
     * @return el movimiento a realizar
     */
    public Movimiento calcularMovimiento(final Tablero tablero) {

        final double[] outputByteArray = test(tablero);

        final List<KeyDoubleBean<Integer>> listIndexCualFicha = DeepLearningUtils.getListArgMaxToMinFromOffsetUntilLength(outputByteArray,
                KnightRawLearningUnit.NUM_MOVIMIENTOS, outputByteArray.length);
        final List<KeyDoubleBean<Integer>> listIndexPosicionFinal = DeepLearningUtils.getListArgMaxToMinUntilLength(outputByteArray,
                KnightRawLearningUnit.NUM_MOVIMIENTOS);
        final List<Casilla> listCasillaFicha = tablero.getCasillasConFichaDeBandoYTipo(this.bando, TipoFicha.CABALLO);

        for (final KeyDoubleBean<Integer> indexFicha : listIndexCualFicha) {

            final int index = indexFicha.getKey() - KnightRawLearningUnit.NUM_MOVIMIENTOS;
            AppLogger.logDebug("Probando index " + index + " de un tamaño " + listCasillaFicha.size());

            if (index >= listCasillaFicha.size()) {
                continue;
            }

            final PosicionTablero posicionOrigen = listCasillaFicha.get(index).getPosicionTablero();

            AppLogger.logDebug("Probando posicionOrigen " + posicionOrigen);

            final List<PosicionTablero> listPosicionTableroEnL = RepositorioPosicionesTableroEnL
                    .getListaPosicionTableroPosicionesEnL(posicionOrigen.getHorizontal(), posicionOrigen.getVertical());
            final Movimiento movimiento = getMovimientoParaOrigenDado(tablero, listPosicionTableroEnL, listIndexPosicionFinal,
                    posicionOrigen);

            if (movimiento != null) {
                return movimiento;
            }
        }

        // No era posible ningun movimiento
        return null;
    }

}
