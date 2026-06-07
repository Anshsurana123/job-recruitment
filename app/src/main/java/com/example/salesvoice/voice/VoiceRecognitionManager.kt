package com.example.salesvoice.voice

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.concurrent.Executors

class VoiceRecognitionManager(private val context: Context) {

    private var recognizer: SpeechRecognizer? = null
    var onResult: ((String) -> Unit)? = null
    var onError: ((Int) -> Unit)? = null
    var onOfflineMissing: (() -> Unit)? = null

    enum class GoogleSpeechState {
        OK,
        DISABLED,
        NOT_INSTALLED,
        SERVICE_UNAVAILABLE
    }

    fun getGoogleSpeechState(): GoogleSpeechState {
        val pm = context.packageManager
        
        // 1. Check if speech recognition is available at all
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            return GoogleSpeechState.SERVICE_UNAVAILABLE
        }
        
        // 2. Query all available RecognitionServices
        val intent = Intent("android.speech.RecognitionService")
        val services = pm.queryIntentServices(intent, 0)
        
        val googlePackages = listOf(
            "com.google.android.googlequicksearchbox",
            "com.google.android.tts"
        )
        
        // Check if any Google speech service is installed
        var anyGoogleInstalled = false
        var anyGoogleEnabled = false
        
        for (pkg in googlePackages) {
            val isInstalled = services.any { it.serviceInfo.packageName == pkg }
            if (isInstalled) {
                anyGoogleInstalled = true
                val appInfo = try {
                    pm.getApplicationInfo(pkg, 0)
                } catch (e: Exception) {
                    null
                }
                if (appInfo != null && appInfo.enabled) {
                    anyGoogleEnabled = true
                }
            }
        }
        
        // Fallback check for any other package starting with com.google.android.
        if (!anyGoogleInstalled) {
            val fallbackMatch = services.firstOrNull { 
                it.serviceInfo.packageName.startsWith("com.google.android.")
            }
            if (fallbackMatch != null) {
                anyGoogleInstalled = true
                val pkg = fallbackMatch.serviceInfo.packageName
                val appInfo = try {
                    pm.getApplicationInfo(pkg, 0)
                } catch (e: Exception) {
                    null
                }
                if (appInfo != null && appInfo.enabled) {
                    anyGoogleEnabled = true
                }
            }
        }
        
        if (!anyGoogleInstalled) {
            return GoogleSpeechState.NOT_INSTALLED
        }
        
        if (!anyGoogleEnabled) {
            return GoogleSpeechState.DISABLED
        }
        
        return GoogleSpeechState.OK
    }

    fun getDisabledPackageName(): String {
        val pm = context.packageManager
        val intent = Intent("android.speech.RecognitionService")
        val services = pm.queryIntentServices(intent, 0)
        
        val googlePackages = listOf(
            "com.google.android.googlequicksearchbox",
            "com.google.android.tts"
        )
        
        // If there's an installed but disabled one, return its package name
        for (pkg in googlePackages) {
            val isInstalled = services.any { it.serviceInfo.packageName == pkg }
            if (isInstalled) {
                val appInfo = try {
                    pm.getApplicationInfo(pkg, 0)
                } catch (e: Exception) {
                    null
                }
                if (appInfo != null && !appInfo.enabled) {
                    return pkg
                }
            }
        }
        
        // Check fallback
        val fallbackMatch = services.firstOrNull { 
            it.serviceInfo.packageName.startsWith("com.google.android.")
        }
        if (fallbackMatch != null) {
            val pkg = fallbackMatch.serviceInfo.packageName
            val appInfo = try {
                pm.getApplicationInfo(pkg, 0)
            } catch (e: Exception) {
                null
            }
            if (appInfo != null && !appInfo.enabled) {
                return pkg
            }
        }
        
        return "com.google.android.googlequicksearchbox"
    }

    private fun getGoogleSpeechRecognizerComponent(context: Context): ComponentName? {
        val pm = context.packageManager
        val intent = Intent("android.speech.RecognitionService")
        val services = pm.queryIntentServices(intent, 0)
        
        val googlePackages = listOf(
            "com.google.android.googlequicksearchbox",
            "com.google.android.tts"
        )
        
        // 1. Try preferred packages in order
        for (pkg in googlePackages) {
            val match = services.firstOrNull { it.serviceInfo.packageName == pkg }
            if (match != null) {
                val appInfo = try {
                    pm.getApplicationInfo(pkg, 0)
                } catch (e: Exception) {
                    null
                }
                if (appInfo != null && appInfo.enabled) {
                    return ComponentName(match.serviceInfo.packageName, match.serviceInfo.name)
                }
            }
        }
        
        // 2. Fallback to any service whose package starts with com.google.android.
        val fallbackMatch = services.firstOrNull { 
            it.serviceInfo.packageName.startsWith("com.google.android.")
        }
        if (fallbackMatch != null) {
            val pkg = fallbackMatch.serviceInfo.packageName
            val appInfo = try {
                pm.getApplicationInfo(pkg, 0)
            } catch (e: Exception) {
                null
            }
            if (appInfo != null && appInfo.enabled) {
                return ComponentName(fallbackMatch.serviceInfo.packageName, fallbackMatch.serviceInfo.name)
            }
        }
        
        return null
    }

    fun checkAndDownloadLanguagePack(languagePreference: String = "en-IN", onDownloadTriggered: (() -> Unit)? = null) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, languagePreference)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, languagePreference)
            }

            val component = getGoogleSpeechRecognizerComponent(context)
            val checkRecognizer = if (component != null) {
                SpeechRecognizer.createSpeechRecognizer(context, component)
            } else {
                SpeechRecognizer.createSpeechRecognizer(context)
            }

            checkRecognizer.checkRecognitionSupport(
                intent,
                Executors.newSingleThreadExecutor(),
                object : RecognitionSupportCallback {
                    override fun onSupportResult(recognitionSupport: RecognitionSupport) {
                        val installedLangs = recognitionSupport.installedOnDeviceLanguages
                        val supportedLangs = recognitionSupport.supportedOnDeviceLanguages
                        val pendingLangs = recognitionSupport.pendingOnDeviceLanguages

                        // Normalization helper
                        fun normalize(lang: String?): String = 
                            lang?.replace('_', '-')?.trim()?.lowercase(java.util.Locale.US).orEmpty()

                        val cleanPreference = normalize(languagePreference)
                        val basePreference = cleanPreference.split('-')[0]

                        // Check if exact or base language match is installed
                        val isInstalled = installedLangs.any {
                            val cleanIt = normalize(it)
                            val baseIt = cleanIt.split('-')[0]
                            cleanIt == cleanPreference || (basePreference.isNotEmpty() && baseIt == basePreference)
                        }

                        // Check if download is already in progress
                        val isPending = pendingLangs.any {
                            val cleanIt = normalize(it)
                            val baseIt = cleanIt.split('-')[0]
                            cleanIt == cleanPreference || (basePreference.isNotEmpty() && baseIt == basePreference)
                        }

                        if (!isInstalled && !isPending) {
                            val isSupported = supportedLangs.any {
                                val cleanIt = normalize(it)
                                val baseIt = cleanIt.split('-')[0]
                                cleanIt == cleanPreference || (basePreference.isNotEmpty() && baseIt == basePreference)
                            }
                            if (isSupported) {
                                Handler(Looper.getMainLooper()).post {
                                    try {
                                        checkRecognizer.triggerModelDownload(intent)
                                        onDownloadTriggered?.invoke()
                                    } catch (e: Exception) {
                                        e.printStackTrace()
                                    }
                                }
                            }
                        }

                        Handler(Looper.getMainLooper()).post {
                            checkRecognizer.destroy()
                        }
                    }

                    override fun onError(error: Int) {
                        Handler(Looper.getMainLooper()).post {
                            checkRecognizer.destroy()
                        }
                    }
                }
            )
        }
    }

    fun startListening(languagePreference: String = "en-IN") {
        destroy() // Reset any previous recognizer session

        val component = getGoogleSpeechRecognizerComponent(context)
        recognizer = if (component != null) {
            SpeechRecognizer.createSpeechRecognizer(context, component)
        } else {
            SpeechRecognizer.createSpeechRecognizer(context)
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languagePreference)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, languagePreference)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)            // Force offline mode
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        recognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onResults(bundle: Bundle) {
                val results = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val transcript = results?.firstOrNull() ?: ""
                onResult?.invoke(transcript)
            }

            override fun onError(errorCode: Int) {
                if (errorCode == SpeechRecognizer.ERROR_CLIENT) {
                    onOfflineMissing?.invoke()
                } else {
                    onError?.invoke(errorCode)
                }
            }

            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        try {
            recognizer?.startListening(intent)
        } catch (e: Exception) {
            e.printStackTrace()
            onError?.invoke(5) // SpeechRecognizer.ERROR_CLIENT
        }
    }

    fun stopListening() {
        recognizer?.stopListening()
    }

    fun destroy() {
        recognizer?.stopListening()
        recognizer?.destroy()
        recognizer = null
    }
}
