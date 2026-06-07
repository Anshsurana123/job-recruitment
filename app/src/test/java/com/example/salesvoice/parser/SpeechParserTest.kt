package com.example.salesvoice.parser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SpeechParserTest {

    @Test
    fun testParseStandardInputs() {
        val testCases = listOf(
            TestCase("2 kg rice", 2.0, "kg", "rice"),
            TestCase("do kilo chawal", 2.0, "kg", "chawal"),
            TestCase("teen litre tel", 3.0, "litre", "tel"),
            TestCase("aadha kg sugar", 0.5, "kg", "sugar"),
            TestCase("five piece bread", 5.0, "piece", "bread"),
            TestCase("1.5 kg atta", 1.5, "kg", "atta"),
            TestCase("paanch sau gram dal", 500.0, "g", "dal"),
            TestCase("dedh kilo maida", 1.5, "kg", "maida"),
            TestCase("rice 2 kg", 2.0, "kg", "rice"),
            TestCase("10 packet namak", 10.0, "packet", "namak"),
            // Hindi / Marathi Devanagari test cases
            TestCase("दो किलो चावल", 2.0, "kg", "चावल"),
            TestCase("आधा लीटर तेल", 0.5, "litre", "तेल"),
            TestCase("डेढ़ किलो चीनी", 1.5, "kg", "चीनी"),
            TestCase("5 पीस ब्रेड", 5.0, "piece", "ब्रेड"),
            TestCase("पाँच सौ ग्राम दाल", 500.0, "g", "दाल"),
            TestCase("chawal 2 किलो", 2.0, "kg", "chawal"),
            TestCase("10 पैकेट नमक", 10.0, "packet", "नमक")
        )

        for (case in testCases) {
            val result = SpeechParser.parse(case.spoken)
            assertNull("Error should be null for phrase: ${case.spoken}", result.error)
            assertEquals("Quantity mismatch for phrase: ${case.spoken}", case.expectedQty, result.quantity!!, 0.01)
            assertEquals("Unit mismatch for phrase: ${case.spoken}", case.expectedUnit, result.unit)
            assertEquals("Product mismatch for phrase: ${case.spoken}", case.expectedProduct, result.productNameSpoken)
        }
    }

    private data class TestCase(
        val spoken: String,
        val expectedQty: Double,
        val expectedUnit: String?,
        val expectedProduct: String
    )
}
