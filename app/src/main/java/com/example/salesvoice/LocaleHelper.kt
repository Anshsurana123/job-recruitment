package com.example.salesvoice

import android.content.Context
import android.content.res.Configuration
import android.os.Build
import java.util.Locale

/**
 * Utility to persist and apply the user's chosen app language.
 * Supports: "en" (English), "hi" (Hindi), "mr" (Marathi).
 */
object LocaleHelper {

    private const val PREFS_NAME = "SalesVoicePrefs"
    private const val KEY_APP_LANGUAGE = "pref_app_language"
    private const val DEFAULT_LANGUAGE = "en"

    /**
     * Returns the persisted app language code ("en", "hi", or "mr").
     */
    fun getLanguage(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_APP_LANGUAGE, DEFAULT_LANGUAGE) ?: DEFAULT_LANGUAGE
    }

    /**
     * Persists the chosen language code.
     */
    fun setLanguage(context: Context, languageCode: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_APP_LANGUAGE, languageCode).apply()
    }

    /**
     * Wraps the given [baseContext] with the persisted locale so that
     * all getString() / layout inflations use the correct language.
     */
    fun applyLocale(baseContext: Context): Context {
        val langCode = getLanguage(baseContext)
        return updateResources(baseContext, langCode)
    }

    private fun updateResources(context: Context, languageCode: String): Context {
        val locale = Locale(languageCode)
        Locale.setDefault(locale)

        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        config.setLayoutDirection(locale)

        return context.createConfigurationContext(config)
    }
}
