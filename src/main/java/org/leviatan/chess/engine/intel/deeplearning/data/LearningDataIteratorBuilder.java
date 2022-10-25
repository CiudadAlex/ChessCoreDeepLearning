package org.leviatan.chess.engine.intel.deeplearning.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.tools.platform.Function3Arg;

/**
 * LearningDataIteratorBuilder.
 *
 * @author Alejandro
 *
 */
public final class LearningDataIteratorBuilder {

    private LearningDataIteratorBuilder() {
    }

    /**
     * Builds the Iterator<LearningData>.
     *
     * @param <T>
     *            tipo de LearningUnit
     * @param partida
     *            partida
     * @param builderLearningUnit
     *            builderLearningUnit
     * @param bando
     *            bando
     * @return Iterator<LearningData>
     */
    public static <T extends LearningUnit> Iterator<LearningData> build(final Partida partida,
            final Function3Arg<Tablero, Movimiento, Bando, T> builderLearningUnit, final Bando bando) {

        final Iterator<Movimiento> iterMovimiento = partida.getIteratorMovimiento();
        final Tablero tablero = new Tablero();

        final List<LearningData> listLearningData = new ArrayList<LearningData>();

        Bando bandoTurno = Bando.BLANCO;

        while (iterMovimiento.hasNext()) {

            final Movimiento movimiento = iterMovimiento.next();

            if (bandoTurno.equals(bando)) {

                // Calcular arrays inmediatamente para evitar problemas de
                // modificacion del tablero
                final T learningUnit = builderLearningUnit.apply(tablero, movimiento, bando);
                final LearningData learningData = learningUnit.buildLearningData();

                if (learningData != null) {
                    // Solo entra las learning data correctas para cada ficha
                    listLearningData.add(learningData);
                }
            }

            tablero.realizarMovimiento(movimiento);
            bandoTurno = bandoTurno.getBandoOpuesto();
        }

        // AppLogger.logDebug("Encontradas muestras: " +
        // listLearningData.size());

        return listLearningData.iterator();
    }

}
