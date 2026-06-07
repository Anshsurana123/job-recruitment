package com.example.salesvoice

import android.content.Context
import androidx.appcompat.app.AppCompatActivity

/**
 * Base activity that applies the user's chosen locale
 * before the content view is created.
 * All activities in SalesVoice should extend this class.
 */
open class BaseActivity : AppCompatActivity() {

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(LocaleHelper.applyLocale(newBase))
    }
}
