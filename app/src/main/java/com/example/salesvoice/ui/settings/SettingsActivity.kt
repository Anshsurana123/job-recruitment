package com.example.salesvoice.ui.settings

import android.app.AlertDialog
import android.content.Context
import android.os.Bundle
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.salesvoice.BaseActivity
import com.example.salesvoice.LocaleHelper
import com.example.salesvoice.R
import com.example.salesvoice.databinding.ActivitySettingsBinding

class SettingsActivity : BaseActivity() {

    private lateinit var binding: ActivitySettingsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        loadAndDisplaySettings()
        setupClickListeners()
    }

    private fun setupToolbar() {
        binding.btnBack.setOnClickListener {
            finish()
        }
    }

    private fun loadAndDisplaySettings() {
        val sharedPrefs = getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)

        val shopName = sharedPrefs.getString("pref_shop_name", "My Shop")
        val currency = sharedPrefs.getString("pref_currency", "₹")
        val languageCode = sharedPrefs.getString("pref_language", "en-IN")

        binding.tvCurrentShopName.text = shopName
        binding.tvCurrentCurrency.text = currency
        
        binding.tvCurrentLanguage.text = when (languageCode) {
            "en-IN" -> "English (India)"
            "hi-IN" -> "Hindi (India)"
            "en-US" -> "English (US)"
            else -> "English (India)"
        }

        val appLang = LocaleHelper.getLanguage(this)
        binding.tvCurrentAppLanguage.text = when (appLang) {
            "hi" -> getString(R.string.lang_hindi)
            "mr" -> getString(R.string.lang_marathi)
            else -> getString(R.string.lang_english)
        }
    }

    private fun setupClickListeners() {
        val sharedPrefs = getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)

        // Shop name click
        binding.cvShopName.setOnClickListener {
            val editText = EditText(this).apply {
                setText(sharedPrefs.getString("pref_shop_name", "My Shop"))
                setSelection(text.length)
            }
            val container = FrameLayout(this).apply {
                val params = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    leftMargin = 50
                    rightMargin = 50
                    topMargin = 20
                    bottomMargin = 20
                }
                addView(editText, params)
            }

            AlertDialog.Builder(this, com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
                .setTitle(getString(R.string.settings_change_shop_name))
                .setView(container)
                .setPositiveButton(getString(R.string.save)) { _, _ ->
                    val newName = editText.text.toString().trim()
                    if (newName.isNotEmpty()) {
                        sharedPrefs.edit().putString("pref_shop_name", newName).apply()
                        loadAndDisplaySettings()
                        Toast.makeText(this, getString(R.string.settings_shop_name_saved), Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this, getString(R.string.settings_shop_name_empty), Toast.LENGTH_SHORT).show()
                    }
                }
                .setNegativeButton(getString(R.string.cancel), null)
                .show()
        }

        // Currency Symbol click
        binding.cvCurrency.setOnClickListener {
            val symbols = arrayOf("₹", "$", "£", "€", "¥")
            val currentSymbol = sharedPrefs.getString("pref_currency", "₹")
            val checkedItem = symbols.indexOf(currentSymbol)

            AlertDialog.Builder(this, com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
                .setTitle(getString(R.string.settings_choose_currency))
                .setSingleChoiceItems(symbols, checkedItem) { dialog, which ->
                    val selectedSymbol = symbols[which]
                    sharedPrefs.edit().putString("pref_currency", selectedSymbol).apply()
                    loadAndDisplaySettings()
                    dialog.dismiss()
                    Toast.makeText(this, getString(R.string.settings_currency_updated, selectedSymbol), Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton(getString(R.string.cancel), null)
                .show()
        }

        // Speech Language click
        binding.cvLanguage.setOnClickListener {
            val languages = arrayOf("English (India) - en-IN", "Hindi (India) - hi-IN", "English (US) - en-US")
            val codes = arrayOf("en-IN", "hi-IN", "en-US")
            val currentCode = sharedPrefs.getString("pref_language", "en-IN")
            val checkedItem = codes.indexOf(currentCode)

            AlertDialog.Builder(this, com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
                .setTitle(getString(R.string.settings_choose_speech_lang))
                .setSingleChoiceItems(languages, checkedItem) { dialog, which ->
                    val selectedCode = codes[which]
                    sharedPrefs.edit().putString("pref_language", selectedCode).apply()
                    loadAndDisplaySettings()
                    dialog.dismiss()
                    Toast.makeText(this, getString(R.string.settings_speech_lang_updated), Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton(getString(R.string.cancel), null)
                .show()
        }

        // App Language click
        binding.cvAppLanguage.setOnClickListener {
            val languages = arrayOf(getString(R.string.lang_english), getString(R.string.lang_hindi), getString(R.string.lang_marathi))
            val codes = arrayOf("en", "hi", "mr")
            val currentCode = LocaleHelper.getLanguage(this)
            val checkedItem = codes.indexOf(currentCode).coerceAtLeast(0)

            AlertDialog.Builder(this, com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
                .setTitle(getString(R.string.settings_choose_app_lang))
                .setSingleChoiceItems(languages, checkedItem) { dialog, which ->
                    val selectedCode = codes[which]
                    LocaleHelper.setLanguage(this, selectedCode)
                    dialog.dismiss()
                    Toast.makeText(this, getString(R.string.settings_app_lang_updated), Toast.LENGTH_SHORT).show()
                    recreate()
                }
                .setNegativeButton(getString(R.string.cancel), null)
                .show()
        }
    }
}
