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
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits.KingRawLearningUnit;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.structure.RawNetStructure;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTablero;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroAlrededor;
import org.leviatan.chess.tools.platform.AppLogger;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.KeyDoubleBean;

/**
 * KingRawNetwork.
 *
 * @author Alejandro
 *
 */
public class KingRawNetwork extends AbstractRawNetwork<KingRawLearningUnit> {

    /**
     * Constructor for KingRawNetwork.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathDirStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public KingRawNetwork(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {
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
        return KingRawLearningUnit.NUM_OUTPUTS;
    }

    @Override
    protected Function3Arg<Tablero, Movimiento, Bando, KingRawLearningUnit> getBuilderLearningUnit() {
        return (tablero, movimiento, bando) -> new KingRawLearningUnit(tablero, movimiento, bando);
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

        PosicionTablero posicionOrigen = null;

        final List<Casilla> listCasilla = tablero.getCasillasConFichaDeBandoYTipo(this.bando, TipoFicha.REY);

        if (listCasilla.isEmpty()) {
            // Error no hay reyes
            return null;

        } else {
            posicionOrigen = listCasilla.get(0).getPosicionTablero();
        }

        final List<PosicionTablero> listaPosicionTableroAlrededor = RepositorioPosicionesTableroAlrededor
                .getListaPosicionTableroPosicionesAlrededor(posicionOrigen.getHorizontal(), posicionOrigen.getVertical());

        final List<KeyDoubleBean<Integer>> listIndexValue = DeepLearningUtils.getListArgMaxToMin(outputByteArray);

        for (final KeyDoubleBean<Integer> indexDoubleBean : listIndexValue) {

            final int index = indexDoubleBean.getKey();

            final PosicionTablero posicionDestino;

            if (index < KingRawLearningUnit.NUM_MOVIMIENTOS_ALREDEDOR) {
                posicionDestino = listaPosicionTableroAlrededor.get(index);

            } else {

                // enroques
                if (outputByteArray[KingRawLearningUnit.NUM_MOVIMIENTOS_ALREDEDOR] > outputByteArray[KingRawLearningUnit.NUM_MOVIMIENTOS_ALREDEDOR
                        + 1]) {
                    // Salto hacia la izquierda
                    posicionDestino = RepositorioPosicionesTablero.generaNuevaPosicionMovida(posicionOrigen.getHorizontal(),
                            posicionOrigen.getVertical(), -2, 0);
                } else {
                    // Salto hacia la derecha
                    posicionDestino = RepositorioPosicionesTablero.generaNuevaPosicionMovida(posicionOrigen.getHorizontal(),
                            posicionOrigen.getVertical(), 2, 0);
                }
            }

            AppLogger.logDebug("Probando posicionOrigen " + posicionOrigen);

            final boolean esMovimientoPosible = esMovimientoPosible(tablero, posicionOrigen, posicionDestino);

            if (esMovimientoPosible) {
                return new Movimiento(posicionOrigen, posicionDestino);
            }
        }

        // No era posible ningun movimiento
        return null;
    }
}
