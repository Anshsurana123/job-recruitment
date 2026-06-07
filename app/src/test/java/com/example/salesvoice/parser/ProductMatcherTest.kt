package com.example.salesvoice.parser

import com.example.salesvoice.data.model.Product
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ProductMatcherTest {

    private val mockProducts = listOf(
        Product(1, "Rice", "chawal, basmati, biryani", "kg", 80.0, 10.0),
        Product(2, "Sugar", "chini, shakar", "kg", 45.0, 5.0),
        Product(3, "Mustard Oil", "sarso tel, oil, tel", "litre", 180.0, 20.0)
    )

    @Test
    fun testFindMatchExact() {
        val match = ProductMatcher.findMatch("rice", mockProducts)
        assertNotNull(match)
        assertEquals(1L, match!!.id)
        assertEquals("Rice", match.name)
    }

    @Test
    fun testFindMatchAlias() {
        val match = ProductMatcher.findMatch("chawal", mockProducts)
        assertNotNull(match)
        assertEquals(1L, match!!.id)

        val match2 = ProductMatcher.findMatch("sarso tel", mockProducts)
        assertNotNull(match2)
        assertEquals(3L, match2!!.id)
    }

    @Test
    fun testFindMatchSubstring() {
        val match = ProductMatcher.findMatch("mustard", mockProducts)
        assertNotNull(match)
        assertEquals(3L, match!!.id)

        val match2 = ProductMatcher.findMatch("chini", mockProducts)
        assertNotNull(match2)
        assertEquals(2L, match2!!.id)
    }

    @Test
    fun testFindMatchPhonetic() {
        val hindiCatalog = listOf(
            Product(4, "चावल", "basmati", "kg", 80.0, 10.0),
            Product(5, "Sugar", "चीनी, shakar", "kg", 45.0, 5.0),
            Product(6, "तांदूळ", "modak", "kg", 90.0, 15.0)
        )

        // 1. Spoken latin matching typed Devanagari product name
        val match1 = ProductMatcher.findMatch("chawal", hindiCatalog)
        assertNotNull(match1)
        assertEquals(4L, match1!!.id)

        // 2. Spoken Devanagari matching typed Devanagari product alias
        val match2 = ProductMatcher.findMatch("चीनी", hindiCatalog)
        assertNotNull(match2)
        assertEquals(5L, match2!!.id)

        // 3. Spoken latin matching typed Devanagari product name
        val match3 = ProductMatcher.findMatch("tandul", hindiCatalog)
        assertNotNull(match3)
        assertEquals(6L, match3!!.id)
    }

    @Test
    fun testFindMatchNoMatch() {
        val match = ProductMatcher.findMatch("flour", mockProducts)
        assertNull(match)
    }
}
