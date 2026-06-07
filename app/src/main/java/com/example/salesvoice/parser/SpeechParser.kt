package com.example.salesvoice.parser

object SpeechParser {

    // NUMBER WORDS → numeric value
    // Covers English and Hindi/Marathi numbers
    private val numberWords = mapOf(
        // English
        "half" to 0.5, "quarter" to 0.25,
        "one" to 1.0, "two" to 2.0, "three" to 3.0, "four" to 4.0,
        "five" to 5.0, "six" to 6.0, "seven" to 7.0, "eight" to 8.0,
        "nine" to 9.0, "ten" to 10.0, "eleven" to 11.0, "twelve" to 12.0,
        "fifteen" to 15.0, "twenty" to 20.0, "twenty five" to 25.0,
        "thirty" to 30.0, "forty" to 40.0, "fifty" to 50.0, "hundred" to 100.0,
        // Hindi (Latin Script)
        "ek" to 1.0, "do" to 2.0, "teen" to 3.0, "char" to 4.0,
        "paanch" to 5.0, "chhe" to 6.0, "saat" to 7.0, "aath" to 8.0,
        "nau" to 9.0, "das" to 10.0, "paanch sau" to 500.0,
        "aadha" to 0.5, "dedh" to 1.5, "dhaayi" to 2.5,
        "bees" to 20.0, "bis" to 20.0, "pachis" to 25.0, "pachhees" to 25.0,
        "tees" to 30.0, "chalis" to 40.0, "pachas" to 50.0,
        // Marathi (Latin Script)
        "ek" to 1.0, "don" to 2.0, "teen" to 3.0, "char" to 4.0,
        "paach" to 5.0, "saha" to 6.0, "saat" to 7.0, "aath" to 8.0,
        "nav" to 9.0, "daha" to 10.0, "ardha" to 0.5,
        "vees" to 20.0, "vis" to 20.0, "panchvees" to 25.0,
        "tees" to 30.0, "chalis" to 40.0, "pannas" to 50.0,
        // Hindi/Marathi (Devanagari Script)
        "एक" to 1.0, "दो" to 2.0, "तीन" to 3.0, "चार" to 4.0,
        "पांच" to 5.0, "पाँच" to 5.0, "छह" to 6.0, "सात" to 7.0,
        "आठ" to 8.0, "नौ" to 9.0, "दस" to 10.0, "पाच" to 5.0,
        "सहा" to 6.0, "नऊ" to 9.0, "दहा" to 10.0,
        "आधा" to 0.5, "अर्धा" to 0.5, "डेढ़" to 1.5, "ढाई" to 2.5,
        "सौ" to 100.0,
        "बीस" to 20.0, "वीस" to 20.0, "पच्चीस" to 25.0, "पंचवीस" to 25.0,
        "तीस" to 30.0, "चालीस" to 40.0, "पचास" to 50.0, "पन्नास" to 50.0,
        // Devanagari two-word number phrases
        "पाँच सौ" to 500.0, "पांच सौ" to 500.0
    )

    // UNIT ALIASES
    private val unitAliases = mapOf(
        "kg" to listOf("kg", "kilo", "kilogram", "kilograms", "किलो", "किग्रा"),
        "g" to listOf("g", "gram", "grams", "grm", "soo", "ग्राम", "ग्रैम"),
        "litre" to listOf("litre", "liter", "litres", "liters", "l", "ltr", "tel", "लीटर", "ली"),
        "piece" to listOf("piece", "pieces", "pcs", "pc", "nos", "number", "nag", "नग", "पीस"),
        "dozen" to listOf("dozen", "dozens", "darjan", "दर्जन"),
        "packet" to listOf("packet", "packets", "pack", "pkt", "पैकेट")
    )

    data class ParseResult(
        val quantity: Double?,
        val unit: String?,
        val productNameSpoken: String?,
        val error: String? = null
    )

    fun parse(transcript: String): ParseResult {
        val rawTokens = transcript.lowercase().trim().split("\\s+".toRegex()).filter { it.isNotBlank() }
        
        val tokens = mutableListOf<String>()
        val tokenRegex = Regex("^([0-9]+(?:\\.[0-9]+)?)([a-zA-Z\\u0900-\\u097F]+)$")
        for (rawToken in rawTokens) {
            val token = rawToken.trim(',', '.', '!', '?', ':', '-')
            val match = tokenRegex.matchEntire(token)
            if (match != null) {
                tokens.add(match.groupValues[1])
                tokens.add(match.groupValues[2])
            } else {
                tokens.add(token)
            }
        }

        // Step 1: Find quantity
        var quantity: Double? = null
        var quantityEndIndex = -1

        // Try direct numeric first (e.g. "2", "1.5", "2.5")
        for ((i, token) in tokens.withIndex()) {
            val num = token.toDoubleOrNull()
            if (num != null) {
                quantity = num
                quantityEndIndex = i
                break
            }
        }

        // Try two-word number phrases (e.g. "twenty five", "paanch sau")
        if (quantity == null) {
            for (i in 0 until tokens.size - 1) {
                val phrase = "${tokens[i]} ${tokens[i+1]}"
                if (numberWords.containsKey(phrase)) {
                    quantity = numberWords[phrase]
                    quantityEndIndex = i + 1
                    break
                }
            }
        }

        // Try single word number (e.g. "do", "teen", "aadha")
        if (quantity == null) {
            for ((i, token) in tokens.withIndex()) {
                if (numberWords.containsKey(token)) {
                    quantity = numberWords[token]
                    quantityEndIndex = i
                    break
                }
            }
        }

        if (quantity == null) return ParseResult(null, null, null, "Could not find a quantity in: \"$transcript\"")

        // Step 2: Find unit (search tokens after quantity)
        var unit: String? = null
        var unitEndIndex = -1

        for (i in (quantityEndIndex + 1) until tokens.size) {
            val token = tokens[i]
            for ((standardUnit, aliases) in unitAliases) {
                if (aliases.contains(token)) {
                    unit = standardUnit
                    unitEndIndex = i
                    break
                }
            }
            if (unit != null) break
        }

        // Step 3: Remaining tokens = product name spoken
        val startIndex = if (unitEndIndex > -1) unitEndIndex + 1 else quantityEndIndex + 1
        var productTokens = tokens.subList(startIndex, tokens.size)

        if (productTokens.isEmpty()) {
            // Maybe product name came before quantity? Try reverse parse.
            // e.g. "rice 2 kg" — take tokens before quantity index
            val before = tokens.subList(0, if (quantityEndIndex > -1) quantityEndIndex else tokens.size)
            // If unit came before quantity, filter out the unit token too (e.g. "rice 2 kg" where quantityEndIndex is 1 and unitEndIndex is 2)
            // Wait, let's see if there is any unit in the prefix. Normally unit comes after quantity, so "before" is just the product name.
            if (before.isNotEmpty()) {
                return ParseResult(quantity, unit, before.joinToString(" "))
            }
            return ParseResult(null, null, null, "Could not find product name in: \"$transcript\"")
        }

        // Clean up remaining filler words (e.g., "of", "kilo of rice" -> product name should be "rice")
        var spokenProduct = productTokens.joinToString(" ")
        if (spokenProduct.startsWith("of ")) {
            spokenProduct = spokenProduct.substring(3).trim()
        }

        return ParseResult(
            quantity = quantity,
            unit = unit,
            productNameSpoken = spokenProduct
        )
    }
}
