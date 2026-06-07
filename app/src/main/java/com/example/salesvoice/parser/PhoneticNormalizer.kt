package com.example.salesvoice.parser

object PhoneticNormalizer {

    // Multi-codepoint Devanagari sequences (checked FIRST during transliteration)
    // Nukta consonants: base + \u093C, and conjuncts: consonant + halant + consonant
    private val multiCharMappings = listOf(
        "क्ष" to "ksh", "ज्ञ" to "gy",
        "ड़" to "d", "ढ़" to "dh", "फ़" to "f", "ज़" to "z", "ख़" to "kh", "ग़" to "g"
    )

    // Single Devanagari character → Latin mapping
    private val singleCharMap = mapOf(
        // Consonants
        'क' to "k", 'ख' to "kh", 'ग' to "g", 'घ' to "gh", 'ङ' to "ng",
        'च' to "ch", 'छ' to "chh", 'ज' to "j", 'झ' to "jh", 'ञ' to "ny",
        'ट' to "t", 'ठ' to "th", 'ड' to "d", 'ढ' to "dh", 'ण' to "n",
        'त' to "t", 'थ' to "th", 'द' to "d", 'ध' to "dh", 'न' to "n",
        'प' to "p", 'फ' to "ph", 'ब' to "b", 'भ' to "bh", 'म' to "m",
        'य' to "y", 'र' to "r", 'ल' to "l", 'व' to "v", 'श' to "sh",
        'ष' to "sh", 'स' to "s", 'ह' to "h", 'ळ' to "l",

        // Vowels (independent)
        'अ' to "a", 'आ' to "a", 'इ' to "i", 'ई' to "i", 'उ' to "u", 'ऊ' to "u",
        'ऋ' to "ri", 'ए' to "e", 'ऐ' to "ai", 'ओ' to "o", 'औ' to "au",

        // Matras (dependent vowel signs)
        'ा' to "a", 'ि' to "i", 'ी' to "i", 'ु' to "u", 'ू' to "u",
        'ृ' to "ri", 'े' to "e", 'ै' to "ai", 'ो' to "o", 'ौ' to "au",
        'ं' to "n", 'ः' to "h", 'ँ' to "n"
    )

    // Characters that are consonants (for implicit 'a' insertion logic)
    private val consonants = setOf(
        'क', 'ख', 'ग', 'घ', 'ङ',
        'च', 'छ', 'ज', 'झ', 'ञ',
        'ट', 'ठ', 'ड', 'ढ', 'ण',
        'त', 'थ', 'द', 'ध', 'न',
        'प', 'फ', 'ब', 'भ', 'म',
        'य', 'र', 'ल', 'व', 'श',
        'ष', 'स', 'ह', 'ळ'
    )

    // Characters that suppress the implicit 'a' after a consonant
    private val matraOrHalant = setOf(
        'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ',
        'ं', 'ः', 'ँ', '्',
        '\u093C' // nukta
    )

    // Common phonetic commodity spelling equivalences
    private val phoneticEquivalents = mapOf(
        "chaval" to "chawal",
        "cheenee" to "chini",
        "cheeni" to "chini",
        "shakkar" to "sugar",
        "shakar" to "sugar",
        "chini" to "sugar",
        "aata" to "atta",
        "aatta" to "atta",
        "tandool" to "tandul",
        "taandool" to "tandul",
        "taandul" to "tandul",
        "doodh" to "dudh",
        "tel" to "oil",
        "tup" to "ghee"
    )

    fun normalize(input: String): String {
        val lower = input.lowercase().trim()
        if (lower.isEmpty()) return ""

        val sb = StringBuilder()
        var i = 0
        while (i < lower.length) {
            // Check multi-char sequences first (greedy match)
            var matched = false
            for ((seq, latin) in multiCharMappings) {
                if (lower.startsWith(seq, i)) {
                    sb.append(latin)
                    i += seq.length
                    // Check if the last char of the sequence was a consonant base
                    // and the next char is NOT a matra/halant → insert implicit 'a'
                    if (i < lower.length && !matraOrHalant.contains(lower[i])) {
                        sb.append("a")
                    }
                    matched = true
                    break
                }
            }
            if (matched) continue

            val char = lower[i]
            val latin = singleCharMap[char]
            if (latin != null) {
                sb.append(latin)
                if (consonants.contains(char) && i + 1 < lower.length) {
                    val nextChar = lower[i + 1]
                    if (!matraOrHalant.contains(nextChar)) {
                        sb.append("a")
                    }
                }
            } else {
                // Skip halant (virama) and nukta — they are consumed by logic above
                if (char != '्' && char != '\u093C') {
                    sb.append(char)
                }
            }
            i++
        }

        var result = sb.toString().replace("  ", " ").trim()

        // Apply common phonetic equivalents
        for ((key, value) in phoneticEquivalents) {
            if (result.contains(key)) {
                result = result.replace(key, value)
            }
        }

        return result
    }
}
