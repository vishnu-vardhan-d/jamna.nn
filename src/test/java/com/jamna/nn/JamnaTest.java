package com.jamna.nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;

/**
 * Regression test: the shipped trained model (jamna.bin) must still sort
 * fresh random sequences with high accuracy. Runs from the project root, where
 * jamna.bin lives. If you change the architecture, retrain and recommit the model.
 */
public class JamnaTest {

    @Test
    public void shippedModelSortsAccurately() throws IOException {
        assertTrue(Jamna.load("jamna.bin"),
            "jamna.bin must be present at the project root");
        double acc = Jamna.accuracy(300);
        assertTrue(acc >= 0.90,
            "expected >= 90% exact-match sort accuracy, got " + Math.round(acc * 100) + "%");
    }
}
