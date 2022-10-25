package org.leviatan.chess.engine.intel.deeplearning.networks.raw.learningunits;

import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Casilla;
import org.leviatan.chess.board.Ficha;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.PosicionTablero;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.engine.deeplearning.DenseNetworkInputGenerator;
import org.leviatan.chess.engine.deeplearning.NetworkInputGenerator;
import org.leviatan.chess.engine.intel.deeplearning.data.CheckFichaRuntimeException;
import org.leviatan.chess.engine.intel.deeplearning.data.LearningUnit;
import org.leviatan.chess.engine.movimientosposibles.Direccion;
import org.leviatan.chess.engine.repositorios.RepositorioPosicionesTableroDireccion;

/**
 * AbstractRawLearningUnit.
 *
 * @author Alejandro
 *
 */
public abstract class AbstractRawLearningUnit implements LearningUnit {

    /** NUM_INPUTS_TABLERO. */
    public static final int NUM_INPUTS_TABLERO = (TipoFicha.values().length + Bando.values().length + 1) * Tablero.TALLA_TABLERO
            * Tablero.TALLA_TABLERO;

    protected final Tablero tablero;
    protected final Movimiento movimiento;
    protected final Bando bando;

    /**
     * Constructor for AbstractRawLearningUnit.
     *
     * @param tablero
     *            tablero
     * @param movimiento
     *            movimiento
     * @param bando
     *            bando
     */
    public AbstractRawLearningUnit(final Tablero tablero, final Movimiento movimiento, final Bando bando) {
        this.tablero = tablero;
        this.movimiento = movimiento;
        this.bando = bando;
    }

    @Override
    public double[] getInput() {
        return getInput(this.tablero);
    }

    /**
     * Devuelve el input de un tablero.
     *
     * @param tablero
     *            tablero
     * @return el input de un tablero
     */
    public static double[] getInput(final Tablero tablero) {

        final NetworkInputGenerator networkInputGenerator = new DenseNetworkInputGenerator();
        return networkInputGenerator.getInput(tablero);
    }

    protected Ficha checkFicha(final TipoFicha tipoFicha) {

        final Ficha ficha = this.tablero.getFicha(this.movimiento.getPosicionOrigen());

        if (!ficha.getTipoFicha().equals(tipoFicha)) {
            throw new CheckFichaRuntimeException("La ficha no es un " + tipoFicha.name().toLowerCase());
        }

        if (!ficha.getBando().equals(this.bando)) {
            throw new CheckFichaRuntimeException("La ficha no es del bando " + this.bando.name().toLowerCase());
        }

        return ficha;
    }

    protected Integer getIndexFicha(final Ficha ficha, final PosicionTablero posicionTablero) {

        final List<Casilla> listCasilla = this.tablero.getCasillasConFichaDeBandoYTipo(ficha.getBando(), ficha.getTipoFicha());
        int index = 0;

        for (final Casilla casilla : listCasilla) {

            if (casilla.getPosicionTablero().equals(posicionTablero)) {
                return index;
            }

            index++;
        }

        return null;
    }

    protected Integer getNumeroMovimientoEnDireccion(final PosicionTablero posicionTableroOrigen,
            final PosicionTablero posicionTableroDestino, final Direccion direccion) {

        final List<PosicionTablero> listPosicionTableroDireccion = RepositorioPosicionesTableroDireccion
                .getListaPosicionTableroPosicionesDireccion(posicionTableroOrigen.getHorizontal(), posicionTableroOrigen.getVertical(),
                        direccion);

        int index = 0;

        for (final PosicionTablero posicionTableroDireccion : listPosicionTableroDireccion) {

            if (posicionTableroDestino.equals(posicionTableroDireccion)) {
                return index;
            }

            index++;
        }

        return null;
    }

    protected void fillInicioOutputConDireccionYIntensidad(final double[] output, final PosicionTablero posicionTableroOrigen,
            final PosicionTablero posicionTableroDestino, final List<Direccion> listDireccion) {

        int index = 0;

        for (final Direccion direccion : listDireccion) {

            final Integer numMovimiento = getNumeroMovimientoEnDireccion(posicionTableroOrigen, posicionTableroDestino, direccion);

            if (numMovimiento != null) {
                output[index] = 1;
                output[listDireccion.size() + numMovimiento] = 1;
                break;
            }

            index++;
        }
    }

}
