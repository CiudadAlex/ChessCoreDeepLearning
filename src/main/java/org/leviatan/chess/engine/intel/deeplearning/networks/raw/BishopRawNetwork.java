package org.leviatan.chess.engine.intel.deeplearning.networks.raw;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Casilla;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits.BishopRawLearningUnit;
import org.leviatan.chess.engine.intel.deeplearning.networks.raw.structure.RawNetStructure;
import org.leviatan.chess.engine.movimientosposibles.Direccion;
import org.leviatan.chess.tools.platform.Function3Arg;
import org.leviatan.chess.tools.platform.KeyDoubleBean;

/**
 * BishopRawNetwork.
 *
 * @author Alejandro
 *
 */
public class BishopRawNetwork extends AbstractRawNetwork<BishopRawLearningUnit> {

    /**
     * Constructor for BishopRawNetwork.
     *
     * @param bando
     *            bando
     * @param pathDirStoreLoad
     *            pathDirStoreLoad
     * @param load
     *            load
     * @throws IOException
     */
    public BishopRawNetwork(final Bando bando, final String pathDirStoreLoad, final boolean load) throws IOException {
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
        return BishopRawLearningUnit.NUM_OUTPUTS;
    }

    @Override
    protected Function3Arg<Tablero, Movimiento, Bando, BishopRawLearningUnit> getBuilderLearningUnit() {
        return (tablero, movimiento, bando) -> new BishopRawLearningUnit(tablero, movimiento, bando);
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
                BishopRawLearningUnit.NUM_DIRECCIONES + BishopRawLearningUnit.NUM_INTENSIDADES, outputByteArray.length);
        final List<KeyDoubleBean<Integer>> listIndexDireccion = DeepLearningUtils.getListArgMaxToMinUntilLength(outputByteArray,
                BishopRawLearningUnit.NUM_DIRECCIONES);
        final List<KeyDoubleBean<Integer>> listIndexIntensidad = DeepLearningUtils.getListArgMaxToMinFromOffsetUntilLength(outputByteArray,
                BishopRawLearningUnit.NUM_DIRECCIONES, BishopRawLearningUnit.NUM_DIRECCIONES + BishopRawLearningUnit.NUM_INTENSIDADES);

        final List<Direccion> listDireccion = Direccion.getSubconjuntoDirecciones(true);

        final List<Casilla> listCasillaFicha = tablero.getCasillasConFichaDeBandoYTipo(this.bando, TipoFicha.ALFIL);

        return getMovimientoDireccion(tablero, listIndexCualFicha, listIndexDireccion, listIndexIntensidad, listDireccion, listCasillaFicha,
                BishopRawLearningUnit.NUM_DIRECCIONES + BishopRawLearningUnit.NUM_INTENSIDADES);
    }
}
