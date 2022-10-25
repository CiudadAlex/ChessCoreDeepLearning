package org.leviatan.chess.engine.intel.deeplearning.networks.raw;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits.WhichPieceRawLearningUnit;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.structure.RawNetStructure;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.KeyDoubleBean;

/**
 * WhichPieceRawNetwork.
 *
 * @author Alejandro
 *
 */
public class WhichPieceRawNetwork extends AbstractRawNetwork<WhichPieceRawLearningUnit> {

    /**
     * Constructor for WhichPieceRawNetwork.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathDirStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public WhichPieceRawNetwork(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {
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
        return WhichPieceRawLearningUnit.NUM_OUTPUTS;
    }

    @Override
    protected Function3Arg<Tablero, Movimiento, Bando, WhichPieceRawLearningUnit> getBuilderLearningUnit() {
        return (tablero, movimiento, bando) -> new WhichPieceRawLearningUnit(tablero, movimiento, bando);
    }

    /**
     * Calcula el tipo de ficha a mover por orden de prioridad.
     *
     * @param tablero
     *            tablero
     * @return el tipo de ficha a mover por orden de prioridad
     */
    public List<TipoFicha> calcularTipoFicha(final Tablero tablero) {

        final double[] outputByteArray = test(tablero);

        final List<KeyDoubleBean<Integer>> listIndexValue = DeepLearningUtils.getListArgMaxToMin(outputByteArray);
        final TipoFicha[] arrayTipoFicha = TipoFicha.values();

        final List<TipoFicha> listTipoFicha = new ArrayList<TipoFicha>();

        for (final KeyDoubleBean<Integer> indexValue : listIndexValue) {
            listTipoFicha.add(arrayTipoFicha[indexValue.getKey()]);
        }

        return listTipoFicha;
    }
}
