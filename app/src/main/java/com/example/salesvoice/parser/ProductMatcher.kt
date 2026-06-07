package com.example.salesvoice.parser

import com.example.salesvoice.data.model.Product

object ProductMatcher {

    fun findMatch(spokenName: String, products: List<Product>): Product? {
        val spoken = spokenName.lowercase().trim()
        if (spoken.isEmpty()) return null

        // 1. Exact name match
        products.find { it.name.lowercase().trim() == spoken }?.let { return it }

        // 2. Alias match (exact)
        products.find { product ->
            product.getAliasList().any { alias -> alias.lowercase() == spoken }
        }?.let { return it }

        // 3. Name contains spoken
        products.find { it.name.lowercase().contains(spoken) }?.let { return it }

        // 4. Spoken contains name
        products.find { spoken.contains(it.name.lowercase()) }?.let { return it }

        // 5. Any alias contains spoken or spoken contains alias
        products.find { product ->
            product.getAliasList().any { alias ->
                val lowerAlias = alias.lowercase()
                lowerAlias.contains(spoken) || spoken.contains(lowerAlias)
            }
        }?.let { return it }

        // 6. Phonetic Matches (Devanagari script / Latin phonetic normalization)
        val normalizedSpoken = PhoneticNormalizer.normalize(spoken)
        if (normalizedSpoken.isNotEmpty()) {
            // Exact normalized name match
            products.find { PhoneticNormalizer.normalize(it.name) == normalizedSpoken }?.let { return it }

            // Exact normalized alias match
            products.find { product ->
                product.getAliasList().any { alias -> PhoneticNormalizer.normalize(alias) == normalizedSpoken }
            }?.let { return it }

            // Substring normalized matches
            products.find { 
                val normName = PhoneticNormalizer.normalize(it.name)
                normName.contains(normalizedSpoken) || normalizedSpoken.contains(normName)
            }?.let { return it }

            products.find { product ->
                product.getAliasList().any { alias ->
                    val normAlias = PhoneticNormalizer.normalize(alias)
                    normAlias.contains(normalizedSpoken) || normalizedSpoken.contains(normAlias)
                }
            }?.let { return it }
        }

        return null
    }
}
